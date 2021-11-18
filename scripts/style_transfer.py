import argparse
import glob
import os
from time import time
import sys
from typing import List, Tuple
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPNN import PNN, GPNN
from utils.image import save_image


def main(contents_and_styles: List[Tuple[str, str]], out_dir: str, coarse_dim: int, double_run=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    PNN_moduel = PNN(patch_size=7, stride=1, alpha=0.005, reduce_memory_footprint=True)
    GPNN_module = GPNN(PNN_moduel, scale_factor=(1, 1), resize=256, num_steps=10, pyr_factor=0.75, coarse_dim=coarse_dim,
                       noise_sigma=0, device=device)

    for (content_image_path, style_image_path) in contents_and_styles:
        content_fname, ext = os.path.splitext(os.path.basename(content_image_path))[:2]
        style_fname, _ = os.path.splitext(os.path.basename(style_image_path))[:2]
        start = time()
        output_image = GPNN_module.run(target_img_path=style_image_path, init_mode=content_image_path)
        img_name = os.path.join(out_dir, f'{GPNN_module.resize}x{GPNN_module.pyr_factor}->{GPNN_module.coarse_dim}', f"{content_fname}-to-{style_fname}{ext}")
        save_image(output_image, img_name)
        if double_run:
            output_image = GPNN_module.run(target_img_path=style_image_path, init_mode=img_name)
            img_name = os.path.join(out_dir, f'{GPNN_module.resize}x{GPNN_module.pyr_factor}->{GPNN_module.coarse_dim}',f"{content_fname}-to-{style_fname}-2-{ext}")
            save_image(output_image, img_name)
        print(f"{content_fname} {style_fname} took {time() - start} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_imgs_directory', type=str, required=True)
    parser.add_argument('--content_imgs_directory', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="output/style_transfer")
    parser.add_argument('--coarse_dim', type=int, default=128)
    parser.add_argument('--double_run', action='store_true', default=False)
    args = parser.parse_args()

    style_imgs = glob.glob(f"{args.style_imgs_directory}/*.jpg")
    content_imgs = glob.glob(f"{args.content_imgs_directory}/*.jpg")

    print(f"Style imgs: {len(style_imgs)}")
    print(f"Content imgs: {len(content_imgs)}")

    contents_and_styles = []
    for style_img in style_imgs:
        contents_and_styles.extend([(content_img, style_img) for content_img in content_imgs])

    print(f"Total number of images: {len(contents_and_styles)}")
    main(contents_and_styles, args.out_dir, args.coarse_dim, args.double_run)
