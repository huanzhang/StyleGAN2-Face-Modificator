import os
from PIL import Image

source_config = [
    (140, 239, 86.5),
    (138.1, 262, 109.5),
    (139.9, 248.5, 96),
    (142.9, 229.6, 77.1),
    (145.5, 213.7, 61.2),
    (148.4, 200.5, 48),
    (150.9, 190.6, 38.1),
    (153.3, 184.4, 31.9),
    (154.9, 181.8, 29.3),
    (154.5, 184.8, 32.3),
    (151.6, 196.2, 43.7),
]


def convert_params(x):
    resize = (x[1]-x[2])*2+50
    detaX = x[0] - resize/2
    detaY = x[2]
    return round(resize), round(detaX), round(detaY)


def main():
    params = list(map(convert_params, source_config))
    back_images = os.listdir("data/images")
    back_images.sort()
    front_images = os.listdir("results/3param")
    front_images.sort()
    for i in range(len(params)):
        p = params[i]
        im1 = Image.open("data/images/" + back_images[i])
        im2 = Image.open("results/3param/" + front_images[i])
        back_im = im1.copy()
        resized_im2 = im2.resize((p[0], p[0]), Image.ANTIALIAS)
        back_im.paste(resized_im2, (p[1], p[2]), resized_im2)
        back_im.save("results/dst/"+back_images[i])


if __name__ == "__main__":
    main()
