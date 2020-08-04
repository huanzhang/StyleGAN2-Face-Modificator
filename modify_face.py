import dnnlib.tflib as tflib
import pretrained_networks
import dnnlib
import os
import sys
import math
import pickle
import imageio
import PIL.Image
import numpy as np
from PIL import Image
import tensorflow as tf
import moviepy.editor as mpy
import matplotlib.pyplot as plt
from IPython.display import clear_output
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

vgg16 = 'vgg16_zhang_perceptual.pkl'
model = 'stylegan2-ffhq-config-f.pkl'
networks_urls = {
    'european': ['https://drive.google.com/uc?id=1--kh2Em5U1qh-H7Lin9FzppkZCQ18c4W', 'generator_model-stylegan2-config-f.pkl'],
    'asian': ['https://drive.google.com/uc?id=1-3XU6KzIVywFoKXx2zG1hW8mH4OYpyO9', 'generator_yellow-stylegan2-config-f.pkl'],
    'asian beauty': ['https://drive.google.com/uc?id=1-04v78_pI59M0IvhcKxsm3YhK2-plnbj', 'generator_star-stylegan2-config-f.pkl'],
    'baby': ['https://drive.google.com/uc?id=1--684mANXSgC3aDhLc7lPM7OBHWuVRXa', 'generator_baby-stylegan2-config-f.pkl']}
network = 'asian'
network_path = 'networks/other/' + networks_urls[network][1]
network_pkl = 'networks/other/' + networks_urls[network][1]

resolution = "512"  # @param [128, 256, 512, 1024]
size = int(resolution), int(resolution)


def move_latent_and_save_3_param(latent_vector, direction_intensity, frame_num, Gs_network, Gs_syn_kwargs):
    os.makedirs('results/3param', exist_ok=True)
    new_latent_vector = latent_vector.copy()
    new_latent_vector[0][:8] = (latent_vector[0] + direction_intensity)[:8]
    images = Gs_network.components.synthesis.run(
        new_latent_vector, **Gs_syn_kwargs)
    result = PIL.Image.fromarray(images[0], 'RGB')
    result.thumbnail(size, PIL.Image.ANTIALIAS)
    result.save('results/3param/result_(' +
                str(frame_num+1000) + ')_3param.png')
    return result


def main(npy_file):
    tflib.init_tf()

    with open(network_pkl, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    w_avg = Gs_network.get_var('dlatent_avg')
    noise_vars = [var for name, var in Gs_network.components.synthesis.vars.items(
    ) if name.startswith('noise')]
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1
    truncation_psi = 0.5

    v = np.load(npy_file)
    v = np.array([v])

    frames = 15  # frames

    parameter1 = 'eye_eyebrow_distance'
    direction_file1 = np.load('latent_directions/' + parameter1 + '.npy')
    intensity1 = 20
    coeffs1 = []
    for i in range(0, frames):
        coeffs1.append(round((i*intensity1)/frames, 3))
    coeffs1 = coeffs1 + list(reversed(coeffs1)) + \
        coeffs1 + list(reversed(coeffs1)) + coeffs1 + \
        list(reversed(coeffs1)) + \
        coeffs1 + list(reversed(coeffs1)) + coeffs1 + \
        list(reversed(coeffs1))

    parameter2 = 'emotion_happy'
    direction_file2 = np.load('latent_directions/' + parameter2 + '.npy')
    intensity2 = 11
    coeffs2 = []
    for i in range(0, frames):
        coeffs2.append(round((i*intensity2)/frames, 3))
    coeffs2 = coeffs2 + list(reversed(coeffs2)) + \
        coeffs2 + list(reversed(coeffs2)) + coeffs2 + \
        list(reversed(coeffs2)) + \
        coeffs2 + list(reversed(coeffs2)) + coeffs2 + \
        list(reversed(coeffs2))

    parameter3 = 'smile'
    direction_file3 = np.load('latent_directions/' + parameter3 + '.npy')
    intensity3 = 11
    coeffs3 = []
    for i in range(0, frames):
        coeffs3.append(round((i*intensity3)/frames, 3))
    coeffs3 = coeffs3 + list(reversed(coeffs3)) + \
        coeffs3 + list(reversed(coeffs3)) + coeffs3 + \
        list(reversed(coeffs3)) + \
        coeffs3 + list(reversed(coeffs3)) + coeffs3 + \
        list(reversed(coeffs3))

    for i in range(frames*5):
        direction_intensity1 = direction_file1 * coeffs1[i]
        direction_intensity2 = direction_file2 * coeffs2[i]
        direction_intensity3 = direction_file3 * coeffs3[i]
        direction_intensity = direction_intensity1 + \
            direction_intensity2 + direction_intensity3
        move_latent_and_save_3_param(
            v, direction_intensity, i, Gs_network, Gs_syn_kwargs)

    # face_img = []
    # img = os.listdir("results/3param")
    # img.sort()
    # print('Animation is created. Please wait.')
    # for i in img:
    #     face_img.append(imageio.imread("results/3param/"+i))
    # face_img = np.array(face_img)
    # imageio.mimsave("results/3param/3param.mp4", face_img)


if __name__ == "__main__":
    main(sys.argv[1])
