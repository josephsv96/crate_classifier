import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm

seq_img = iaa.Sequential([iaa.Crop(percent=(0, 0.1)),
                          iaa.Fliplr(0.5),
                          iaa.Affine(
                              translate_percent={
                                  "x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                              rotate=(-0.1, 0.1),
                              shear=(-1, 1))],
                         random_order=True,
                         random_state=0)

seq_img_i = seq_img.to_deterministic()


def get_img_augs(img_arr, num_exp=3):
    img_aug = np.zeros(img_arr.shape)
    j = 0
    for i in range(num_exp):
        img_aug[:, :, :, j:j +
                num_exp] = seq_img_i.augment_images(images=img_arr[:, :, :, j:j+num_exp])
        j += num_exp
    return img_aug


def generate_aug(img_arr, num_gen, num_exp):
    """To generate more augmented images
    """
    max_index = img_arr.shape[0]
    img_arr_gen = np.zeros(
        [num_gen, img_arr.shape[1], img_arr.shape[2], img_arr.shape[3]])
    for i in tqdm(range(num_gen)):
        random_index = np.random.randint(0, max_index)
        img_arr_gen[i, :, :, :] = img_arr[random_index, :, :, :]

    img_aug_gen = get_img_augs(img_arr_gen, num_exp)
    return img_arr_gen, img_aug_gen
