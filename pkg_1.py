from tqdm import tqdm
import numpy as np

# Local Modules
import pkg_1a
import pkg_1b
from utils import load_json


def pkg_1(PKG_1_PARAMS):
    # pkg_1a
    img_paths, ann_paths = pkg_1a.data_checker(PKG_1_PARAMS["src_dir"],
                                               PKG_1_PARAMS["num_exp"])

    # pkg_1b
    image_db, annot_db = pkg_1b.sort_by_class(img_paths, ann_paths,
                                              PKG_1_PARAMS)
    # pkg_1c

    return image_db, annot_db


def main(PKG_1_PARAMS=None):
    """
    # pkg_1a
    img_paths, ann_paths = pkg_1a.main(PKG_1_PARAMS)
    # pkg_1b
    image_db, annot_db = sort_by_class(img_paths, ann_paths, PKG_1_PARAMS)
    """
    PKG_1_PARAMS = load_json("pkg_1_config.json")

    image_db, annot_db = pkg_1(PKG_1_PARAMS)
    print("img_paths['class_1'] sample:\n", image_db["class_1"][:4])
    print("ann_paths['class_1'] sample:\n", annot_db["class_1"][:4])

    return None


if __name__ == "__main__":
    main()
