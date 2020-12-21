# Local Modules
import pkg_1a
import pkg_1b
import pkg_1c
import statistics
from utils import load_json


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
    pkg_1a_obj.logging()

    # pkg_1b
    pkg_1b_obj = pkg_1b.DataSorter(pkg_1a_obj.img_paths,
                                   pkg_1a_obj.ann_paths,
                                   PKG_1_PARAMS)
    pkg_1b_obj.logging()

    # Generating augmented data
    pkg_1c_obj = pkg_1c.AugDataGenerator(pkg_1b_obj.image_db,
                                         pkg_1b_obj.annot_db,
                                         PKG_1_PARAMS)
    pkg_1c_obj.logging()

    # TODO: Saving statistics to outdir
    statistics.class_dist_from_db(pkg_1b_obj.annot_db)

    return None


if __name__ == "__main__":
    main()
