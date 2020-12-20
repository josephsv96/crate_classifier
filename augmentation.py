# Dependencies
import numpy as np
from imgaug import augmenters as iaa
from pathlib import Path
from tqdm import tqdm
import cv2

# Local Modules
from utils import create_output_folder
from utils import save_npy_v2
from utils import read_cmp
from utils import get_timestamp


class Augmenter:
    """Class load image sets at different exposures and their annotations
    to generate augmented images from them.
    """

    def __init__(self, PARAMS, img_paths, ann_paths, out_dir=None):

        self.img_paths = img_paths
        self.ann_paths = ann_paths
        self.src_h, self.src_w = PARAMS["img_src_shape"]
        self.out_h, self.out_w = PARAMS["net_in_shape"]

        self.num_exp = PARAMS["num_exp"]
        self.num_img = int(len(img_paths) / self.num_exp)

        self.AUG_CONFIG = PARAMS["augmentation"]
        self.GAUSS_CONFIG = PARAMS["guassian"]

        self.out_dir = out_dir

        if out_dir is None:
            out_path = create_output_folder(Path.cwd(),
                                            folder_name="output_")

            self.out_dir = "/".join(str(out_path).split("\\")
                                    ) + "/" + get_timestamp()
            (Path(self.out_dir)).mkdir()
            (Path(self.out_dir) / "images").mkdir()
            (Path(self.out_dir) / "npy_images").mkdir()
            (Path(self.out_dir) / "npy_annots").mkdir()

    @ staticmethod
    def get_augmenters(num_gen, aug_config):
        """Generate a list of deterministic augmenters. num_gen = 0 or None
        will return a single Augmenter

        Args:
            num_gen (int): Number of augmenters

        Returns:
            list: List of Augmeters
        """
        seq_img = iaa.Sequential([iaa.Fliplr(0.5),
                                  iaa.Affine(scale=aug_config["scale"],
                                             translate_percent={
                                      "x": aug_config["trans_x"],
                                      "y": aug_config["trans_y"]},
            rotate=aug_config["rotate"],
            shear=aug_config["shear"]
        )],
            random_order=False, random_state=0)

        seq_img_list = seq_img.to_deterministic(n=num_gen)

        return seq_img_list

    @ staticmethod
    def to_square(img_instance):
        """Converts a wide image to square image, without loosing resolution.

        Args:
            img_instance (numpy.array): Image in wide resolution

        Returns:
            numpy.array: Squared image
        """
        if len(img_instance.shape) != 3:
            img_instance = np.expand_dims(img_instance, axis=-1)

        if img_instance.shape[1] > img_instance.shape[0]:
            dim = img_instance.shape[0]
            mid = int(img_instance.shape[1]/2)
            img_sq = img_instance[:, int(mid-dim/2):int(mid+dim/2), :]
        else:
            dim = img_instance.shape[1]
            mid = int(img_instance.shape[0]/2)
            img_sq = img_instance[int(mid-dim/2):int(mid+dim/2), :, :]

        return img_sq

    @ staticmethod
    def gaussian_blur(img_instance, gauss_config):
        """Apply Gaussian filter to an image instance

        Args:
            img_instance (numpy.array): Input image array
            gauss_config (dict): Filter parameters

        Returns:
            numpy.array: Blurred image output
        """
        img_instance = cv2.GaussianBlur(img_instance,
                                        ksize=tuple(gauss_config["ksize"]),
                                        sigmaX=gauss_config["sigma"],
                                        borderType=cv2.BORDER_DEFAULT)

        return img_instance

    def get_img_augs(self, img_files, augmenter):
        """Apply augmenter to img_files array.

        Args:
            img_files (numpy.array): Image set; (height, width, num_exp * 3)
            augmenter (imgaug.augmenters.meta.Augmenter): Augmenter

        Returns:
            numpy.array: Augmented image set of (out_h, out_w, num_exp * 3)
        """
        img_aug = np.zeros([self.out_h, self.out_w, self.num_exp*3],
                           dtype=np.float32)

        j = 0
        for i in range(self.num_exp):
            image_instance = cv2.imread(str(img_files[i]),
                                        cv2.IMREAD_UNCHANGED)

            aug_img = augmenter.augment_image(image_instance)

            img_buffer = self.to_square(self.gaussian_blur(aug_img,
                                                           self.GAUSS_CONFIG))
            img_buffer = cv2.resize(img_buffer, (self.out_h, self.out_w),
                                    interpolation=cv2.INTER_NEAREST)
            img_aug[:, :, j:j+3] = img_buffer
            j += 3

        return img_aug

    def get_ann_augs(self, annot, augmenter):
        """Apply augmenter to annotation array.

        Args:
            annot (numpy.array): Annotation; (height, width)
            augmenter (imgaug.augmenters.meta.Augmenter): Augmenter

        Returns:
            numpy.array: Augmented annotation set of (out_h, out_w, 1)
        """
        annot_aug = np.zeros([self.out_h, self.out_w, 1],
                             dtype=np.float32)

        aug_img = augmenter.augment_image(annot[:, :])
        ann_buffer = self.to_square(aug_img)
        annot_aug[:, :, 0] = cv2.resize(ann_buffer, (self.out_h, self.out_w),
                                        interpolation=cv2.INTER_NEAREST)

        return annot_aug

    def generate_aug(self, num_gen, r_state=1, write_img=False, start_index=0):
        """To generate augmented image and annotation sets.
        Same augmentation is applied to a set of images and its annotation.

        Args:
            num_gen (int): Number of images to be generated.
            r_state (int, optional): Random state to select image set.
            Defaults to 1.
            write_img (bool, optional): Flag to write images to output folder.
            Defaults to False.

        Returns:
            numpy.array: Augmented annotation set of (out_h, out_w, num_exp*3)
            numpy.array: Augmented annotation set of (out_h, out_w, 1)
        """
        img_aug_arr = np.zeros(
            [num_gen, self.out_h, self.out_w, self.num_exp*3],
            dtype=np.float32)

        ann_aug_arr = np.zeros(
            [num_gen, self.out_h, self.out_w, 1],
            dtype=np.float32)

        # Generating augmenters
        augs = self.get_augmenters(num_gen=num_gen, aug_config=self.AUG_CONFIG)

        # Select image and corresponding annotation randomly
        # for i in range(num_gen):
        for i in tqdm(range(num_gen)):

            np.random.seed(i * abs(r_state))
            random_index = np.random.randint(0, self.num_img - 1)

            # Select image
            img_index = random_index * self.num_exp
            img_files = self.img_paths[img_index:img_index + self.num_exp]

            # Select annotation
            ann_file = self.ann_paths[random_index]

            # Image Generation
            # (i,128,128,9) = f(random_index,964,1292,9)
            img_aug_arr[i, :, :, :] = self.get_img_augs(img_files,
                                                        augmenter=augs[i])

            # Annot Generation
            # (i,128,128,1) = f(random_index,964,1292,1)
            annot_instance = read_cmp(ann_file, (self.src_h, self.src_w))
            ann_aug_arr[i, :, :, :] = self.get_ann_augs(annot_instance,
                                                        augmenter=augs[i])

            # Saving images and masks
            if write_img is True:
                j = 0
                exp_names = [chr(ord('`')+i+1) for i in range(self.num_exp)]

                # writing images to .bmp
                for _, ch in zip(range(self.num_exp), exp_names):
                    img_name = f"img_{str(i+start_index).zfill(6)}_{ch}.bmp"
                    img_file = f"{self.out_dir}/images/{img_name}"
                    cv2.imwrite(img_file,
                                img_aug_arr[i, :, :, j:j+self.num_exp])
                    j += self.num_exp

                # writing image set to .npy
                img_name = f"img_{str(i+start_index).zfill(6)}"
                img_file = f"{self.out_dir}/npy_images/{img_name}"
                save_npy_v2(img_aug_arr[i, :, :, :], img_file)

                # writing annotation to .npy
                ann_name = f"ann_{str(i+start_index).zfill(6)}"
                ann_file = f"{self.out_dir}/npy_annots/{ann_name}"
                save_npy_v2(ann_aug_arr[i, :, :, :], ann_file)

                # !WORK-IN-PROGRESS
                # writing annotation to .cmp
                # write_cmp(ann_file, ann_aug_arr[i, :, :, :])

        return img_aug_arr, ann_aug_arr
