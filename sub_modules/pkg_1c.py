from pathlib import Path

# Local Modules
from sub_modules import pkg_1a
from sub_modules import pkg_1b
from augmentation import Augmenter
from utils import load_json
from utils import save_npy_v2
from utils import get_timestamp
# from uitls import get_custom_cmap
# from preprocessing import get_dataset


class AugDataGenerator:

    def __init__(self, image_db, annot_db, PARAMS):
        self.image_db = image_db
        self.annot_db = annot_db
        self.PARAMS = PARAMS

        # Defining the output path
        self.output_dir = PARAMS["out_dir"] + "/" + get_timestamp()

        # Creating output folder and subfolders
        AugDataGenerator.mkdir_dataset(self.output_dir)
        self.num_gen = AugDataGenerator.gen_aug_data(image_db,
                                                     annot_db,
                                                     PARAMS,
                                                     self.output_dir)

    @staticmethod
    def mkdir_dataset(output_dir="pkg_1_output"):
        """Create output directory with subfolders

        Args:
            output_dir (str, optional): Output directory. Defaults to
                                        "pkg_1_output".

        """
        (Path(output_dir)).mkdir()
        (Path(output_dir) / "images").mkdir()
        (Path(output_dir) / "npy_images").mkdir()
        (Path(output_dir) / "npy_annots").mkdir()
        (Path(output_dir) / "datasets").mkdir()

        return None

    @staticmethod
    def gen_aug_data(image_db, annot_db, PARAMS, output_dir):
        GEN_CONFIG = PARAMS["aug_gen"]

        class_selected = [f"class_{i}" for i in GEN_CONFIG["class_to_gen"]]

        img_sel = []
        ann_sel = []

        # Add class weightage here
        for img_class in class_selected:
            img_files = image_db[img_class]
            ann_files = annot_db[img_class]

            img_sel += img_files
            ann_sel += ann_files

        dataset_size = GEN_CONFIG["dataset_size"]

        img_chunks = [img_sel[x:x+dataset_size * PARAMS["num_exp"]]
                      for x in range(0, len(img_sel), dataset_size * PARAMS["num_exp"])]

        ann_chunks = [ann_sel[x:x+dataset_size]
                      for x in range(0, len(ann_sel), dataset_size)]

        img_numbering = 0

        for i in range(len(img_chunks)):

            num_gen = int(GEN_CONFIG["gen_scale"] * len(ann_chunks[i]))
            aug_obj = Augmenter(PARAMS, img_chunks[i], ann_chunks[i],
                                output_dir)

            aug_img, aug_ann = aug_obj.generate_aug(num_gen=num_gen,
                                                    r_state=GEN_CONFIG["r_state"],
                                                    write_img=True,
                                                    start_index=img_numbering)

            img_numbering += num_gen

            save_npy_v2(aug_img, Path(output_dir)/f"datasets/images_{i+1}")
            save_npy_v2(aug_ann, Path(output_dir)/f"datasets/annots_{i+1}")

        return img_numbering

    def logging(self):
        """Logging for pkg_1c
        """
        print("Number of augmented images generated:", self.num_gen)

        return None


def main():
    """
    # pkg_1a
    img_paths, ann_paths = pkg_1a.main(PKG_1_PARAMS)
    # pkg_1b
    image_db, annot_db = sort_by_class(img_paths, ann_paths, PKG_1_PARAMS)
    """

    # Initial Parameters
    PKG_1_PARAMS = load_json("pkg_1_config.json")

    # pkg_1a
    pkg_1a_obj = pkg_1a.DataChecker(PKG_1_PARAMS["src_dir"],
                                    PKG_1_PARAMS["num_exp"])
    # pkg_1b
    pkg_1b_obj = pkg_1b.DataSorter(pkg_1a_obj.img_paths,
                                   pkg_1a_obj.ann_paths,
                                   PKG_1_PARAMS)

    # Generating augmented data
    pkg_1c_obj = AugDataGenerator(pkg_1b_obj.image_db,
                                  pkg_1b_obj.annot_db,
                                  PKG_1_PARAMS)

    pkg_1c_obj.logging()
    return None


if __name__ == "__main__":
    main()
