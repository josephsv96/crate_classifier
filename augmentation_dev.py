# Dependencies
import numpy as np
from imgaug import augmenters as iaa
from pathlib import Path
from tqdm import tqdm
import cv2

# Local Modules
from utilities import create_output_folder
from utilities import save_npy_v2
# from utilities import write_cmp


class Augmenter:
    """Class load image sets at different exposures and their annotations and
    to generate augmented images from them.
    """

    def __init__(self, img_arr, ann_arr, out_h, out_w, num_exp, out_dir=None):
        self.img_arr = img_arr
        self.ann_arr = ann_arr
        self.out_h = out_h
        self.out_w = out_w

        self.num_exp = num_exp
        self.num_img = img_arr.shape[0]

        self.out_dir = out_dir

        if out_dir is not None:
            create_output_folder(Path(self.out_dir))

    @staticmethod
    def get_augmenters(num_gen):
        """Generate a list of deterministic augmenters. num_gen = 0 or None
        will return a single Augmenter

        Args:
            num_gen (int): Number of augmenters

        Returns:
            list: List of Augmeters
        """
        # seq_img = iaa.Sequential([iaa.Fliplr(0.5),
        #                           iaa.Affine(scale=(1.0, 1.2),
        #                                      translate_percent={
        #                               "x": (-0.25, 0.07), "y": (-0.01, 0.01)},
        #                               rotate=(-4.0, 4.0), shear=(-0.1, 0.1))],
        #                          random_order=False, random_state=0)

        # No x <-> y movement and shear
        seq_img = iaa.Sequential([iaa.Fliplr(0.5),
                                  iaa.Affine(scale=(1.0, 1.2),
                                             rotate=(-4.0, 4.0))],
                                 random_order=False, random_state=0)

        seq_img_list = seq_img.to_deterministic(n=num_gen)

        return seq_img_list

    @staticmethod
    def to_square(img_instance):
        """Converts a wide image to square image, without loosing resolution.
        Assuming width > height.

        Args:
            img_instance (numpy.array): Image in wide resolution

        Returns:
            numpy.array: Squared image
        """
        if len(img_instance.shape) != 3:
            img_instance = np.expand_dims(img_instance, axis=-1)
        height = img_instance.shape[0]
        cent_x = int(img_instance.shape[0]/2)
        img_sq = img_instance[:, int(cent_x-height/2):int(cent_x+height/2), :]

        return img_sq

    @staticmethod
    def gaussian_blur(img_instance):
        img_instance = cv2.GaussianBlur(
            img_instance, (19, 19), 0.4*11, cv2.BORDER_DEFAULT)
        return img_instance

    def get_img_augs(self, img_set, augmenter):
        """Apply augmenter to img_set array.

        Args:
            img_set (numpy.array): Image set; (height, width, num_exp * 3)
            augmenter (imgaug.augmenters.meta.Augmenter): Augmenter

        Returns:
            numpy.array: Augmented image set of (out_h, out_w, num_exp * 3)
        """
        img_aug = np.zeros([self.out_h, self.out_w, self.num_exp*3],
                           dtype=np.float32)

        j = 0
        for i in range(self.num_exp):
            aug_img = augmenter.augment_image(img_set[:, :, j:j+self.num_exp])
            img_buffer = self.to_square(self.gaussian_blur(aug_img))
            img_buffer = cv2.resize(img_buffer, (self.out_h, self.out_w),
                                    interpolation=cv2.INTER_NEAREST)
            img_aug[:, :, j:j+self.num_exp] = img_buffer
            j += self.num_exp

        return img_aug

    def get_ann_augs(self, annot, augmenter):
        """Apply augmenter to annotation array.

        Args:
            annot (numpy.array): Annotation; (height, width, 1)
            augmenter (imgaug.augmenters.meta.Augmenter): Augmenter

        Returns:
            numpy.array: Augmented annotation set of (out_h, out_w, 1)
        """
        annot_aug = np.zeros([self.out_h, self.out_w, 1],
                             dtype=np.float32)

        aug_img = augmenter.augment_image(annot[:, :, 0])
        annot_aug[:, :, 0] = cv2.resize(self.to_square(aug_img),
                                        (self.out_h, self.out_w))

        return annot_aug

    def generate_aug(self, num_gen, r_state=1, write_img=False):
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
        augs = self.get_augmenters(num_gen=num_gen)

        # Select image and corresponding annotation randomly
        for i in tqdm(range(num_gen)):

            np.random.seed(i * abs(r_state))
            random_index = np.random.randint(0, self.num_img)

            # Select image
            image_instance = self.img_arr[random_index]
            img_aug_arr[i, :, :, :] = self.get_img_augs(image_instance,
                                                        augmenter=augs[i])
            # (i,128,128,9) = f(random_index,964,1292,9)

            # Select annotation
            annot_instance = self.ann_arr[random_index]
            ann_aug_arr[i, :, :, :] = self.get_ann_augs(annot_instance,
                                                        augmenter=augs[i])
            # (i,128,128,1) = f(random_index,964,1292,1)

            # Saving images and masks
            if write_img is True:
                j = 0
                exp_names = [chr(ord('`')+i+1) for i in range(self.num_exp)]

                # writing images to .bmp
                for k, ch in zip(range(self.num_exp), exp_names):
                    img_file = f"{self.out_dir}/img_{str(i).zfill(3)}_{ch}.bmp"
                    cv2.imwrite(img_file,
                                img_aug_arr[i, :, :, j:j+self.num_exp])
                    j += self.num_exp

                # writing annotation to .npy
                ann_file = f"{self.out_dir}/img_{str(i).zfill(3)}_mask"
                save_npy_v2(ann_aug_arr[i, :, :, :], ann_file)

                # !WORK-IN-PROGRESS
                # writing annotation to .cmp
                # write_cmp(ann_file, ann_aug_arr[i, :, :, :])

        return img_aug_arr, ann_aug_arr
