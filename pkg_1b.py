# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
from tqdm import tqdm


# %%
# Local Modules
from utilities import load_json
from utilities import read_cmp
from segmap_v2 import SegmentaionMap
import pkg_1a


# %%
def sort_by_class(img_paths, ann_paths, PKG_1_PARAMS):
    """
    Return sorted dicts of each class by pixel density sorting
    """
    class_init = [f"class_{i}" for i in range(PKG_1_PARAMS["num_classes"])]
    image_db = {class_name: [] for class_name in class_init}
    annot_db = {class_name: [] for class_name in class_init}

    img_id = 0

    for annot_file in tqdm(ann_paths):
        annot_arr = read_cmp(annot_file, PKG_1_PARAMS["img_src_shape"])
        segmap_obj = SegmentaionMap(annot_arr,
                                    PKG_1_PARAMS["num_classes"],
                                    PKG_1_PARAMS["bg_class_id"])
        # fg_cls / activated_px, bg_cls/ total_px
        cls_den_arr = segmap_obj.activation_density()
        # Avoiding background classes
        # cls_den_arr_original = cls_den_arr.copy()
        cls_den_arr[segmap_obj.bg_class_id] = 0
        try:
            detected_cls = np.where(
                cls_den_arr > PKG_1_PARAMS["cls_thres"])[0][0]
        except IndexError:
            # Pushing images with less than threshold to or first id or last id
            # detected_cls = segmap_obj.fg_class_id[-1]
            detected_cls = 0

        annot_db[f"class_{detected_cls}"].append(annot_file)

        # Appending image files
        i = 0
        while(i < PKG_1_PARAMS["num_exp"]):
            image_db[f"class_{detected_cls}"].append(img_paths[img_id + i])
            i += 1
        img_id += PKG_1_PARAMS["num_exp"]

    return image_db, annot_db


# %%
def main(PKG_1_PARAMS=None):
    # "C:/Users/josep/Documents/work/crate_classifier_dev_v2/images/factory_images_annotated"
    # "C:/Users/josep/Documents/work/crate_classifier_dev_v2/images/img_3_sample_train"
    if PKG_1_PARAMS is None:
        PKG_1_PARAMS = load_json("pkg_1_config.json")

    # pkg_1a
    img_paths, ann_paths = pkg_1a.main(PKG_1_PARAMS)
    # pkg_1b
    image_db, annot_db = sort_by_class(img_paths, ann_paths, PKG_1_PARAMS)

    return image_db, annot_db


# %%
if __name__ == "__main__":
    image_db, annot_db = main()
    print("img_paths['class_1'] sample:\n", image_db["class_1"][:4])
    print("ann_paths['class_1'] sample:\n", annot_db["class_1"][:4])


# %%
