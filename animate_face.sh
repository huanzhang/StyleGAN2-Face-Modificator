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

echo "**Wipe background the first time**"
rm -rf ../face-parsing.PyTorch/image/
mkdir ../face-parsing.PyTorch/image/
cp aligned_images/* ../face-parsing.PyTorch/image/
cd ../face-parsing.PyTorch
if command -v deactivate &> /dev/null;then
    deactivate
fi
source .venv/bin/activate
rm -rf res/out/
mkdir res/out/
python wipe_background.py
rm -rf ../StyleGAN2-Face-Modificator/aligned_images/ 
mkdir ../StyleGAN2-Face-Modificator/aligned_images/
cp res/out/*.jpg ../StyleGAN2-Face-Modificator/aligned_images/

echo "**Train neural network**"
cd ../StyleGAN2-Face-Modificator/
if command -v deactivate &> /dev/null;then
    deactivate
fi
source .venv/bin/activate
rm -rf generated_images/
mkdir generated_images/
steps=500
rate=0.1
python project_images.py aligned_images/ generated_images/ --vgg16-pkl 'networks/vgg16_zhang_perceptual.pkl' --num-steps $steps --initial-learning-rate $rate --network-pkl networks/other/generator_yellow-stylegan2-config-f.pkl --video=False

echo "**Animating face modification**"
rm -rf results/
mkdir results/
file_name="${input##*/}"
file="${file_name%.*}"
npy_file="${file}_01.npy"
python modify_face.py "generated_images/$npy_file"


echo "**Wipe background the second time**"
cd ../face-parsing.PyTorch
if command -v deactivate &> /dev/null;then
    deactivate
fi
source .venv/bin/activate

rm -rf image/
mkdir image/
cp ../StyleGAN2-Face-Modificator/results/3param/*.png image/
python wipe_background.py


echo "**Composing source and photo**"
cd ../StyleGAN2-Face-Modificator
if command -v deactivate &> /dev/null;then
    deactivate
fi
source .venv/bin/activate
rm -rf results/3param/
mkdir results/3param/
rm -rf results/dst/
mkdir results/dst/
cp ../face-parsing.PyTorch/res/out/*.png results/3param/
python compose.py
