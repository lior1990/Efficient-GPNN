import argparse
import os
import tempfile
from time import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPNN import PNN, GPNN
from utils.image import save_image
from PIL import Image


def concat_images(images) -> "Image":
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    image_paths = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    images = [Image.open(open(p, "rb")) for p in image_paths]
    new_image = concat_images(images)
    td = tempfile.TemporaryDirectory()
    new_image_path = os.path.join(td.name, "new_image.png")
    new_image.save(new_image_path)

    PNN_moduel = PNN(patch_size=7,
                     stride=1,
                     alpha=0.005,
                     reduce_memory_footprint=True)
    GPNN_module = GPNN(PNN_moduel,
                       scale_factor=(1/len(images), 1/len(images)),
                       resize=0,
                       num_steps=10,
                       pyr_factor=0.75,
                       coarse_dim=14,
                       noise_sigma=0.75,
                       device="cuda:0")

    out_dir = f"outputs/reshuffle_concat/{os.path.basename(dataset_dir)}"
    os.makedirs(out_dir, exist_ok=True)
    im_path = new_image_path
    fname, ext = os.path.splitext(os.path.basename(im_path))[:2]
    for i in range(25):
        start = time()
        output_image = GPNN_module.run(target_img_path=im_path, init_mode="target")
        print(f"took {time() - start} s")
        save_image(output_image, os.path.join(out_dir, f"{fname}_{i}{ext}"))

    td.cleanup()
