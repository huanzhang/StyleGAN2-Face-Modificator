#!/usr/bin/env bash

input="$1"
if [ -f "$input" ]; then
    rm -rf raw_images/
    mkdir raw_images/
    cp $input raw_images/
else
    echo "$input does not exist."
    exit 1
fi


if command -v deactivate &> /dev/null;then
    deactivate
fi
source .venv/bin/activate

echo "**Crop photo**"
rm -rf aligned_images/
mkdir aligned_images/
python align_images.py raw_images/ aligned_images/

echo "**Train neural network**"
rm -rf generated_images/
mkdir generated_images/
python project_images.py aligned_images/ generated_images/ --vgg16-pkl 'networks/vgg16_zhang_perceptual.pkl' --num-steps 1200 --initial-learning-rate 0.02 --network-pkl networks/other/generator_yellow-stylegan2-config-f.pkl --video=False

echo "**Animating face modification**"
rm -rf results/
mkdir results/
file_name="${input##*/}"
file="${file_name%.*}"
npy_file="${file}_01.npy"
python modify_face.py "generated_images/$npy_file"