import numpy as np
from pathlib import Path

# Local Modules
from utils import load_json, rm_duplicate


class DataChecker:
    def __init__(self, src_dir, num_exp):
        self.src_dir = src_dir
        self.num_exp = num_exp

        self.img_paths, self.ann_paths = DataChecker.data_checker(src_dir,
                                                                  num_exp)

    @staticmethod
    def data_checker(src_dir, num_exp, img_ext=".bmp", ann_ext=".cmp"):
        """Checks if there are missing annotation files in the source folder.
        Return pathlib.Path files for images and annotations

        WARNING!! DOES NOT CHECK IF ANNOTATIONS ARE CORRECT (by name)
        ADD CROSS VALIDATION

        Args:
            src_dir (str): Source path
            num_exp (int): Number of exposures of the images
            img_ext (str, optional): Image File Extension. Defaults to ".bmp".
            ann_ext (str, optional): Annotation File Extension. Defaults to
                                    ".cmp".

        Returns:
            list: Valid image file paths
            list: Valid annotation file paths
        """
        src_path = Path(src_dir)

        # All image and annotation paths
        image_paths = list(src_path.glob("**/*" + img_ext))
        annot_paths = list(src_path.glob("**/*" + ann_ext))

        img_nums = rm_duplicate([file.stem.split("_")[1]
                                 for file in image_paths])
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

        valid_image_paths = valid_image_paths
        valid_annot_paths = valid_annot_paths

        return valid_image_paths, valid_annot_paths

    def logging(self):
        """Logging for pkg_1a
        """
        print("img_paths sample:", self.img_paths[:4], sep="\n")
        print("ann_paths sample:", self.ann_paths[:4], sep="\n")

        return None


def main():
    # Initial Parameters
    PKG_1_PARAMS = load_json("pkg_1_config.json")

    # Checking for consistency in the dataset
    pkg_1a_obj = DataChecker(PKG_1_PARAMS["src_dir"],
                             PKG_1_PARAMS["num_exp"])
    pkg_1a_obj.logging()

    return None


if __name__ == "__main__":
    main()
