import argparse
import os
from time import time
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPNN import PNN, GPNN
from utils.image import save_image

device = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_dir', type=str, required=True)
    parser.add_argument('--eval_dataset_dir', type=str, required=True)
    parser.add_argument('--coarse_dim', type=int, default=14)
    parser.add_argument('--noise_sigma', type=float, default=0)
    args = parser.parse_args()

    train_dataset_dir = args.train_dataset_dir
    eval_dataset_dir = args.eval_dataset_dir
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
                       coarse_dim=args.coarse_dim,
                       noise_sigma=args.noise_sigma,  # todo: add noise?
                       device=device)

    out_dir = f"outputs/reshuffle_mixed"
    debug_out_dir = f"outputs/reshuffle_mixed_debug"
    print(f"train image paths: {train_image_paths}")

    for im_path in eval_image_paths:
        print(f"Working on {im_path}")
        fname, ext = os.path.splitext(os.path.basename(im_path))[:2]
        start = time()
        output_image = GPNN_module.run_multiple_images(train_image_paths, im_path, debug_dir=debug_out_dir)
        print(f"took {time() - start} s")
        save_image(output_image, os.path.join(out_dir, f"{fname}${ext}"))
