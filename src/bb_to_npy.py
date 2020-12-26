import numpy as np
from math import ceil
from pathlib import Path
import cv2

from utils import load_txt, save_npy_v2, sort_path


def bb_to_npy(img_file, label):
    src = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
    height, width = src.shape[0], src.shape[1]
    annot = np.zeros([height, width])
    # print(annot.shape)
    for box in label:
        box_class = box[0]

        box_x = box[1] * width
        box_y = box[2] * height
        box_w = box[3] * width
        box_h = box[4] * height
        # print(box_x, box_y, box_w, box_h, box_class)
        x_index_1 = int(box_x - box_w/2)
        x_index_2 = int(box_x + box_w/2)
        y_index_1 = int(box_y - box_h/2)
        y_index_2 = int(box_y + box_h/2)
        (annot[y_index_1:y_index_2, x_index_1:x_index_2]).fill(ceil(box_class))

    return annot


def main():
    # image_dir = Path(input("Enter path of image files:"))
    # label_dir = Path(input("Enter path of bounding box txt files:"))
    # output_dir = Path(input("Output dir:"))

    image_dir = Path(
        "C:/Users/josep/Documents/work/labelling_tools/OpenLabeling/main/input")
    label_dir = Path(
        "C:/Users/josep/Documents/work/labelling_tools/OpenLabeling/main/output/YOLO_darknet")
    output_dir = Path(
        "C:/Users/josep/Documents/work/crate_classifier/dataset/data_1/annots")

    label_arr = load_txt(label_dir)
    bmp_files = sort_path(list(image_dir.glob('**/*.bmp')))

    for i, bmp_file in enumerate(bmp_files):
        label = label_arr[i]
        annot_arr = bb_to_npy(bmp_file, label)
        outfile = output_dir / (bmp_file.stem)
        save_npy_v2(annot_arr, outfile)


if __name__ == "__main__":
    main()
