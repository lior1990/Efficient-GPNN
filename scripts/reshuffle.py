import argparse
import os
from time import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPNN import PNN, GPNN
from utils.image import save_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    image_paths = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
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
                       noise_sigma=0.75,
                       device="cuda:0")

    out_dir = f"outputs/reshuffle/{os.path.basename(dataset_dir)}"
    for im_path in image_paths:
        fname, ext = os.path.splitext(os.path.basename(im_path))[:2]
        for i in range(25):
            start = time()
            output_image = GPNN_module.run(target_img_path=im_path, init_mode="target")
            print(f"took {time() - start} s")
            save_image(output_image, os.path.join(out_dir, f"{fname}_{i}{ext}"))
