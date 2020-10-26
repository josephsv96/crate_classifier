# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
"""
# pkg_1
- pkg_0 -> pkg_1
- Genertes the Dataset for the NN.
## Tasks:
1. Data Loader
- Reads images and annotations from source folder.
2. Dataset Check
- Checks for missing annotations files.
3. Statistics
- Finds the distribution of classes by pixel density.
4. Augmenter
- Genertes augmented images and corressponding annotations.
"""


# %%
import numpy as np
from pathlib import Path

# Local Modules
from utilities import load_json


# %%
def rm_duplicate(seq):
    """
    Remove duplicates from a list preserving order
    https://www.peterbe.com/plog/uniqifiers-benchmark
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


# %%
def run_data_checker(src_dir, num_exp, img_ext=".bmp", ann_ext=".cmp"):
    """
    Checks if there are missing annotation files in the source folder. 
    Return pathlib.Path files for images and annotations
    WARNING!! DOES NOT CHECK IF ANNOTATIONS ARE CORRECT (by name)
    ADD CROSS VALIDATION LATER
    """
    src_path = Path(src_dir)

    # All image and annotation paths
    image_paths = list(src_path.glob("**/*" + img_ext))
    annot_paths = list(src_path.glob("**/*" + ann_ext))

    img_nums = rm_duplicate([file.stem.split("_")[1] for file in image_paths])
    ann_nums = [file.stem.split("_")[1] for file in annot_paths]

    img_dict = dict(enumerate(img_nums))
    indices = {v: k for k, v in img_dict.items()}

    # Matching
    valid_sets = set(img_dict.values()).intersection(ann_nums)
    valid_indices = np.sort([indices[value] for value in valid_sets])

    # Missing annotations
    missing_sets = np.sort(
        list(set(img_dict.values()).symmetric_difference(ann_nums)))
    if missing_sets.size > 0:
        print(f"Missing annotation files: {missing_sets}")

    # Valid paths
    valid_image_paths = []
    valid_annot_paths = []
    for i, index in enumerate(valid_indices):
        j = index * num_exp
        valid_annot_paths.append(annot_paths[i])
        for k in range(num_exp):
            valid_image_paths.append(image_paths[j+k])

    return valid_image_paths, valid_annot_paths


# %%
def main(PKG_1_PARAMS=None):
    # Initial Parmeters
    if PKG_1_PARAMS is None:
        PKG_1_PARAMS = load_json("pkg_1_config.json")
    # pkg 1a
    img_paths, ann_paths = run_data_checker(
        PKG_1_PARAMS["src_dir"], PKG_1_PARAMS["num_exp"])

    return img_paths, ann_paths


# %%
if __name__ == "__main__":
    img_paths, ann_paths = main()
    print("img_paths sample:\n", img_paths[:4])
    print("ann_paths sample:\n", ann_paths[:4])


# %%
