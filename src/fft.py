"""FFT of individual crate classes using sliding window
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_npy
from utilities import load_json, bgr_to_rgb_img
from segmap_v2 import SegmentaionMap

# Local Modules


from sliding import SlidingWindow
import segmap_v2

# Testing Sliding window splitting


def sort_img_arr_by_class(ann_arr, PARAMS):
    sector_name = [f"sec_{i}" for i in range(ann_arr.shape[0])]
    sec_db = {class_name: [] for class_name in sector_name}

    img_id = 0
    for i in range(ann_arr.shape[0]):
        annot = ann_arr[i, :, :, 0]
        segmap_obj = SegmentaionMap(annot,
                                    PARAMS["num_classes"],
                                    PARAMS["bg_class_id"])
        # fg_cls / activated_px, bg_cls/ total_px
        cls_den_arr = segmap_obj.activation_density()
        try:
            detected_cls = np.where(cls_den_arr > PARAMS["det_thres"])[0][0]
        except IndexError:
            # Pushing images with less than threshold to or first id or last id
            detected_cls = 0

        sec_db[f"sec_{i}"] = detected_cls

    return sec_db


def main():
    DATASET_PATH = Path("outputs/images_augmented")
    # DATASET_PATH = Path("C:/Users/josep/Documents/work/crate_classifier_dev/outputs/img_3_sample_valid_centered_small")

    PKG_1_PARAMS = load_json("pkg_1_config.json")

    img_arr = load_npy(DATASET_PATH / "dataset_images_aug.npy") / 255
    ann_arr = load_npy(DATASET_PATH / "dataset_annots_aug.npy")

    index = 5

    # plt.figure(figsize=(10, 8))
    # plt.subplot(1, 2, 1)
    # plt.imshow(bgr_to_rgb_img(img_arr[index, :, :, 3:6]))
    # plt.xticks([])
    # plt.yticks([])

    # plt.subplot(1, 2, 2)
    # plt.imshow(ann_arr[index, :, :, 0])
    # plt.clim([0, 10])
    # plt.xticks([])
    # plt.yticks([])

    # plt.tight_layout()
    # plt.show()

    sample_ann = ann_arr[index, :, :, :]
    my_scale = 2
    sliding_obj = SlidingWindow(sample_ann, w_scale=my_scale,
                                PARAMS=PKG_1_PARAMS)

    # sliding_obj.show_slices(num_class=10)

    print(f"annot_arr.shape: {sample_ann.shape}")
    print(f"N_dim: {sliding_obj.N_dim}")
    print(f"w_dim: {sliding_obj.w_dim}")
    print(f"w_stride: {sliding_obj.w_stride}")
    print(f"sliced_arr.shape: {sliding_obj.sliced_arr.shape}")

    sliced = sliding_obj.sliced_arr
    ind = 1
    plt.imshow(sliced[ind, :, :, 0])
    plt.show()

    annot_db = sort_img_arr_by_class(sliced, PKG_1_PARAMS)

    sliding_obj.show_slices_overlay(num_class=10)

    sliding_obj.image_overlay(img_arr[index, :, :, 3:6])

    np.array(sliding_obj.sort_img_arr_by_class())
    [markdown]
    # ## Sample out of Network

    sample_img = np.load(DATASET_PATH / "output_img.npy")
    sample_ann = np.load(DATASET_PATH / "output_ann.npy")

    my_scale = 4
    sliding_obj = SlidingWindow(
        sample_ann, w_scale=my_scale, PARAMS=PKG_1_PARAMS)

    sliding_obj.show_slices(num_class=10)
