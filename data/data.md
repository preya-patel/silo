# Dataset

This project uses the FFHQ (Flickr-Faces-HQ) face dataset. We trained
on the first 1000 images of the FFHQ-1000 subset and evaluated on
5 held-out images included in the `test_images/` folder of this repo.

## Downloading FFHQ-1000

We obtained FFHQ from a Kaggle mirror. To reproduce our training:

1. Go to https://www.kaggle.com/datasets/dollarakshay/ffhq-1000
   (or search Kaggle for "FFHQ 1000")
2. Sign in to Kaggle (free account required)
3. Click "Download" to grab the dataset as a zip
4. Unzip into a folder of your choice
5. Pass that folder path as `--train_dir` when running training:

   python code/silo_v2.py --mode train \
       --task inpaint \
       --train_dir /path/to/ffhq_1000 \
       --train_steps 2000 \
       --ckpt checkpoints/silo_v2_inpaint.pt

## Image format

FFHQ images are 1024x1024 RGB PNGs. Our pipeline resizes them to
512x512 during training and evaluation.

## Test images

The 5 images in `test_images/` were drawn from the same FFHQ-1000
distribution but excluded from training. We deliberately keep these
in the repo so the reported results are reproducible without
re-downloading the full dataset.
