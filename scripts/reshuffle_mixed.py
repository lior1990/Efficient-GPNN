import os
from time import time
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPNN import PNN, GPNN
from utils.image import save_image

device = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    train_dataset_dir = 'images/2imgs'
    eval_dataset_dir = 'images/eval'
    train_image_paths = [os.path.join(train_dataset_dir, x) for x in os.listdir(train_dataset_dir)]
    eval_image_paths = [os.path.join(eval_dataset_dir, x) for x in os.listdir(eval_dataset_dir)]

    PNN_moduel = PNN(patch_size=7,
                     stride=1,
                     alpha=0.005,
                     reduce_memory_footprint=True)
    GPNN_module = GPNN(PNN_moduel,
                       scale_factor=(1, 1),
                       resize=0,
                       num_steps=10,
                       pyr_factor=0.75,
                       coarse_dim=14,
                       noise_sigma=0,  # todo: add noise?
                       device=device)

    out_dir = f"outputs/reshuffle_mixed"
    for im_path in eval_image_paths:
        fname, ext = os.path.splitext(os.path.basename(im_path))[:2]
        start = time()
        output_image = GPNN_module.run_multiple_images(train_image_paths, im_path)
        print(f"took {time() - start} s")
        save_image(output_image, os.path.join(out_dir, f"{fname}${ext}"))
