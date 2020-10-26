import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm
from cv2 import resize

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


def square_img(img_instance):
    if len(img_instance.shape) != 3:
        img_instance = np.expand_dims(img_instance, axis=-1)
    height = img_instance.shape[0]
    cent_x = int(img_instance.shape[0]/2)
    img_sq = img_instance[:, int(cent_x-height/2):int(cent_x+height/2), :]
    return img_sq


def get_img_augs(img_arr, new_h=128, new_w=128, num_exp=3):
    img_aug = np.zeros([img_arr.shape[0], new_h, new_w, img_arr.shape[-1]],
                       dtype=np.float32)
    print(img_aug.shape)
    j = 0
    for i in range(num_exp):
        aug_img = seq_img_i.augment_images(img_arr[:, :, :, j:j+num_exp])
        for k in range(img_arr.shape[0]):
            img_aug[k, :, :, j:j +
                    num_exp] = resize(square_img(aug_img[k, :, :, :]), (new_h, new_w))
        j += num_exp
    return img_aug


def get_ann_augs(img_arr, new_h=128, new_w=128):
    img_aug = np.zeros([img_arr.shape[0], new_h, new_w, 1],
                       dtype=np.float32)

    aug_img = seq_img_i.augment_images(img_arr[:, :, :, 0])
    for k in range(img_arr.shape[0]):
        img_aug[k, :, :, 0] = resize(
            square_img(aug_img[k, :, :]), (new_h, new_w))

    return img_aug


def generate_aug(img_arr, num_gen, num_exp):
    """To generate more augmented images
    """
    max_index = img_arr.shape[0]
    img_arr_gen = np.zeros([num_gen, img_arr.shape[1], img_arr.shape[2], img_arr.shape[3]],
                           dtype=np.float32)

    for i in tqdm(range(num_gen)):
        np.random.seed(i)
        random_index = np.random.randint(0, max_index)
        img_arr_gen[i, :, :, :] = img_arr[random_index, :, :, :]

    if num_exp != 1:
        img_aug_gen = get_img_augs(img_arr_gen)
    else:
        img_aug_gen = get_ann_augs(img_arr_gen)

    return img_aug_gen
