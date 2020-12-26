"""
To make a dataset from images
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.utils import save_npy_v2, sort_path, resize_arr, read_image, load_npy


def main():
    # image_dir = Path(input("Enter path of image files:"))
    # annot_dir = Path(input("Enter path of bounding box txt files:"))

    image_dir = Path(
        "C:/Users/josep/Documents/work/crate_classifier/dataset/data_1/images")
    annot_dir = Path(
        "C:/Users/josep/Documents/work/crate_classifier/dataset/data_1/annots")
    output_dir = Path(
        "C:/Users/josep/Documents/work/crate_classifier/dataset/data_1/dataset")

    bmp_files = sort_path(list(image_dir.glob('**/*.bmp')))
    annot_files = sort_path(list(annot_dir.glob('**/*.npy')))

    out_width, out_height = [256, 256]
    batch_size = len(bmp_files)

    image_npy_arr = np.zeros([batch_size, out_width, out_height, 3])
    annot_npy_arr = np.zeros([batch_size, out_width, out_height, 1])

    for i in tqdm(range(batch_size)):
        image_npy_arr[i, :, :, :] = resize_arr(
            read_image(bmp_files[i])[:, :, :3], out_width, out_height)
        annot_npy_arr[i, :, :, 0] = resize_arr(
            load_npy(annot_files[i]), out_width, out_height)

    save_npy_v2(image_npy_arr, output_dir / 'data_1_images')
    save_npy_v2(annot_npy_arr, output_dir / 'data_1_labels')


if __name__ == "__main__":
    main()
