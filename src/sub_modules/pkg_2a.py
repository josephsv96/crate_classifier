from keras.utils import Sequence


import os
import cv2
import copy
import threading
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug as ia
from keras.utils import Sequence
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from rhs_utils import KeyPointPair, draw_segmap
import math

import cv2


def read_annotations(img_dir, num_exposures):
    all_imgs = []
    seen_labels = {}
    i_base_file = 0
    i_bmp_file = 0

    all_file_names = sorted(os.listdir(img_dir))
    img = {'image_file_names': [], 'segmap_file_name': []}

    i_in_tuple = 0  # image in tuple counter
    for file_name in all_file_names:
        ext = os.path.splitext(file_name)[1]

        if ext == '.bmp':
            img['image_file_names'].append(img_dir + file_name)
            i_in_tuple += 1

        elif ext == '.cmp':
            img['segmap_file_name'] = img_dir + file_name
            i_in_tuple += 1

        if i_in_tuple > num_exposures:
            all_imgs += [img]
            i_in_tuple = 0  # reset image in tuple counter

            for i in img['image_file_names']:
                print("bmp_file=", i)
            print("cmp_file=", img['segmap_file_name'])
            img = {'image_file_names': [], 'segmap_file_name': []}

    return all_imgs


class YoloBatchGenerator(Sequence):
    def __init__(self,
                 all_imgs,
                 config,
                 shuffle=True,
                 jitter=True,
                 norm=None):
        self.generator = None

        self.all_imgs = all_imgs
        self.config = config

        print("self.config=", self.config)

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.image_counter = 0
        self.lock = threading.Lock()

        ia.seed(1)

        # augmentors by https://github.com/aleju/imgaug

        def sometimes(aug):
            return iaa.Sometimes(0.3, aug)

        self.aug_pipe = iaa.Sequential(
            [
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.05, 0.05),
                                       "y": (-0.05, 0.05)},
                    rotate=(-5, 5),
                    shear=(-0.1, 0.1),
                    mode="edge")
                ),
                iaa.SomeOf(1,
                           [iaa.Add((-10, 10), per_channel=1),
                            iaa.Multiply((0.7, 1.3), per_channel=0)
                            ]
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self.all_imgs)

    # scale augmented exposure-images and segmap downto size of CNN input gate
    def scale_imgs_and_segmap(self, imgs_array_aug, segmap_aug):
        # classmap section
        # print( np.shape( segmap_aug ) )

        scale = self.config['SCALE']
        width_scaled = int(self.config['IMAGE_WDT'] * scale)
        height_scaled = int(self.config['IMAGE_HGT'] * scale)

        segmap_aug_scaled = cv2.resize(
            segmap_aug, dim, interpolation=cv2.INTER_NEAREST)

        plt.imshow(segmap_aug_scaled, interpolation='nearest')
        plt.show()

        imgs_array_aug_scaled = np.zeros(
            (width_scaled, height_scaled, self.config['NUM_CHANNELS']))

        for i_expo in range(self.config['NUM_EXPOSURES']):
            img = cv2.GaussianBlur(imgs_array_aug[:, :, i_expo*self.config['NUM_CHANNELS']:(i_expo+1)*self.config['NUM_CHANNELS']],
                                   (self.config['GAUSS_KERNEL_RADIUS'] * 2+1,
                                    self.config['GAUSS_KERNEL_RADIUS']*2+1),
                                   self.config['GAUSS_SIGMA'] *
                                   self.config['GAUSS_KERNEL_RADIUS'],
                                   cv2.BORDER_DEFAULT)
            img_aug_scaled = cv2.resize(
                img, (width_scaled, height_scaled), interpolation=cv2.INTER_LINEAR)
            plt.imshow(img_aug_scaled, interpolation='nearest')
            plt.show()

            # concatenate images to get a train array with shape( IMAGE_H, IMAGE_W, 3*num_exposures )
            imgs_array_aug_scaled = np.concatenate(
                (imgs_array_aug_scaled, img_aug_scaled), axis=2)

    def __len__(self):
        return int(np.ceil(float(len(self.all_imgs))/self.config['BATCH_SIZE']))

    def size(self):
        return len(self.all_imgs)

    # load original sized segmap file
    def load_segmap(self, segmap_file_name):
        file = open(segmap_file_name, "rb")  # open binary file

        # get file size
        file.seek(0, 2)  # set pointer to end of file
        nb_bytes = file.tell()
        file.seek(0, 0)  # set pointer to begin of file
        buf = file.read(nb_bytes)
        file.close()
        # convert from byte stream to numpy array
        segmap_orig = np.asarray(list(buf), dtype=np.byte)
        segmap_orig = segmap_orig.reshape(
            (self.config['IMAGE_H'], self.config['IMAGE_W']))

        if self.config['CRATES_ONLY'] == 1:
            segmap_orig = np.where(segmap_orig > 0, 1, 0).astype(np.byte)

        return segmap_orig

    def load_images(self, i_images_tuple):
        # read all exposures of this i_images_tuple
        images = []
        for iExposure in range(self.config['NUM_EXPOSURES']):
            images.append(cv2.imread(
                self.all_imgs[i_images_tuple]['image_file_names'][iExposure]))
        return images

    def y_from_segmap(self, segmap):
        nb_cells_x = self.config['GRID_H']
        nb_cells_y = self.config['GRID_W']
        input_wdt = self.config['INPUT_W']
        input_hgt = self.config['INPUT_H']
        y = np.zeros((nb_cells_x, nb_cells_y, self.config['NUM_CLASSES']))

        # old coarse grid
        if False:
            for i_cell_y in range(nb_cells_y):
                start_pix_y = int(1.0*i_cell_y*input_hgt/nb_cells_y)
                end_pix_y = int(1.0*(i_cell_y+1)*input_hgt/nb_cells_y)
                for i_cell_x in range(nb_cells_x):
                    start_pix_x = int(1.0*i_cell_x*input_wdt/nb_cells_x)
                    end_pix_x = int(1.0*(i_cell_x+1)*input_wdt/nb_cells_x)
                    v_nb_class = np.zeros(self.config['NUM_CLASSES'])
                    nb_cell_pix = (end_pix_x-start_pix_x) * \
                        (end_pix_y - start_pix_y)

                    for i_pix_y in range(start_pix_y, end_pix_y):
                        for i_pix_x in range(start_pix_x, end_pix_x):
                            val = segmap[i_pix_y, i_pix_x]
                            v_nb_class[val] += 1

                    for i_class in range(self.config['NUM_CLASSES']):
                        y[i_cell_y, i_cell_x, i_class] = 1.0 * \
                            v_nb_class[i_class]/nb_cell_pix

        # new pixel grid
        for i_cell_y in range(nb_cells_y):
            for i_cell_x in range(nb_cells_x):
                i_class = segmap[i_cell_y, i_cell_x]
                y[i_cell_y, i_cell_x, i_class] = 1

        return y

    def save_augimage(self, nb, path, tag, img):  # save augmented image for debugging
        fullName = path + "img_" + str(nb) + "_"+tag+".bmp"
        cv2.imwrite(fullName, img)

    # save augmented segmap for debugging
    def save_augsegmap(self, nb, path, tag, segmap):
        fullName = "..\\imgaug\\img_" + str(nb) + "_"+tag + ".cmp"
        buf = segmap.tobytes()
        file = open(fullName, "wb")
        file.write(buf)

    def save_y(self, nb, path, y):  # save augmented y for debugging
        fullName = "..\\imgaug\\img_" + str(nb) + ".txt"
        file = open(fullName, "w")
        file.write(np.array_str(y))

    def __getitem__(self, idx):  # returns a complete batch pair x_batch and y_batch
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        x_batch = np.zeros((0))
        y_batch = np.zeros((0))
        x_batch = x_batch.reshape(
            (0, self.config['IMAGE_H'], self.config['IMAGE_W'], 3*self.config['NUM_EXPOSURES']))
        y_batch = y_batch.reshape(
            (0, self.config['GRID_H'], self.config['GRID_W'], self.config['NUM_CLASSES']))
        # old y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], 1 + self.config['NUM_CLASSES']))

        i_img = 0
        images_batch = []
        segmaps_batch = []

        self.lock.acquire()  # to ensure synchronization of image and segmap

        # keypoints-list aufbauen
        num_images = len(self.all_imgs)
        for instance_count in range(r_bound - l_bound):
            instance_src_index = (l_bound + instance_count) % num_images
            # augment all exposure instances in all_imgs
            imgs = self.load_images(instance_src_index)

            # construct output grid fro segment map
            segmap_file_name = self.all_imgs[instance_src_index]['segmap_file_name']
            segmap = self.load_segmap(segmap_file_name)

            # for synchronization, but maybe it´s not necessary
            self.aug_pipe_det = self.aug_pipe.to_deterministic()

            # combine images to a single stacked array
            imgs_array = np.copy(imgs[0])
            for i_img in range(1, self.config['NUM_EXPOSURES']):
                # concatenate images to get a train array with shape( IMAGE_H, IMAGE_W, 3*num_exposures )
                imgs_array = np.concatenate((imgs_array, imgs[i_img]), axis=2)

            segmap = SegmentationMapsOnImage(segmap, shape=imgs_array.shape)
            imgs_array_aug, segmap_aug = self.aug_pipe_det(
                image=imgs_array, segmentation_maps=segmap)  # augment images with 3*num_exposures

            # scale augmented exposure-images and segmap downto size of CNN input gate
            imgs_array_aug_scaled, segmap_aug_scaled = self.scale_imgs_and_segmap(
                imgs_array_aug, segmap_aug)

            y = self.y_from_segmap(segmap_aug_scaled.get_arr())

            #self.save_augimage( self.image_counter, "..\\imgaug\\", "a", imgs_array_aug[...,0:3].reshape((self.config['IMAGE_H'],self.config['IMAGE_W'],3 )) )
            #self.save_augimage( self.image_counter, "..\\imgaug\\", "b", imgs_array_aug[...,3:6].reshape((self.config['IMAGE_H'],self.config['IMAGE_W'],3 )) )
            #self.save_augimage( self.image_counter, "..\\imgaug\\", "c", imgs_array_aug[...,6:9].reshape((self.config['IMAGE_H'],self.config['IMAGE_W'],3 )) )
            #self.save_augsegmap( self.image_counter, "..\\imgaug\\","a",  segmap_aug.get_arr() )
            #self.save_augsegmap( self.image_counter, "..\\imgaug\\","b",  segmap_aug.get_arr() )
            #self.save_augsegmap( self.image_counter, "..\\imgaug\\","c",  segmap_aug.get_arr() )


#            segmap = SegmentationMapsOnImage(segmap, shape=img.shape)

#            self.aug_pipe_det = self.aug_pipe.to_deterministic() # for synchronization, but maybe it´s not necessary

            # augment all images in tuple with same augmentation mask
#            images_aug_i = []


#            for i_img in self.config['num_exposures']
#                image_aug_i, segmap_aug_i = self.aug_pipe_det(image=img, segmentation_maps=segmap) #augment one image in tuple
#                images_aug_i.append( image_aug_i )
#                old_segmap_aug_i = np.copy( segmap_aug_i )
#                if i_img == 0:
#                    train_array = np.copy( image_aug_i )
#                elif i_img > 0:
#                    train_array = np.concatenate( (train_array, image_aug_i), axis=2 ) # concatenate images to get a train array with shape( IMAGE_H, IMAGE_W, 3*num_exposures )
#                    diff = np.sum( segmap_aug_i - old_segmap_aug_i )  # check if all images in tuple are augmented in same manner
#                    print( "augdiff=", diff )

            # merge augmented images in tuple

#            y = self.y_from_segmap( segmaps_aug_i.get_arr() )

            #self.save_augimage( self.image_counter, "..\\imgaug\\", images_aug_i.reshape((self.config['IMAGE_H'],self.config['IMAGE_W'],3 )) )
            #self.save_y( self.image_counter, "..\\imgaug\\", np.argmax( y, axis=2 ) )
            #self.save_augsegmap( self.image_counter, "..\\imgaug\\",  segmaps_aug_i.get_arr() )

            self.image_counter += 1
            i_img += 1

#            x_batch = np.append( x_batch, images_aug_i.reshape((1,self.config['IMAGE_H'],self.config['IMAGE_W'],3 )),0)
            x_batch = np.append(x_batch, imgs_array_aug_scaled.reshape(
                (1, self.config['INPUT_H'], self.config['INPUT_W'], 3*self.config['NUM_EXPOSURES'])), 0)
            y_batch = np.append(y_batch, y.reshape(
                (1, self.config['GRID_H'], self.config['GRID_W'], self.config['NUM_CLASSES'])), 0)

        self.lock.release()
        x_batch = self.norm(x_batch)

        np.set_printoptions(threshold=np.inf, linewidth=500)

        # for i in range( x_batch.shape[0] ):
        #    x = x_batch[i,:]*255.0
        #    self.save_augimage( i, "..\\imgaug\\", x )
        #    self.save_y( i, "..\\imgaug\\", y_batch[i, :] )#np.argmax( y_batch[i,:], axis=2 ) )
        # self.save_y( i, "..\\imgaug\\", x )#np.argmax( y_batch[i,:], axis=2 ) )

        return x_batch, y_batch  # x_batch=image normalized and y_batch=one_hot_classes

    def on_epoch_end(self):
        if self.shuffle:
            # shuffle along the first axis only
            np.random.shuffle(self.all_imgs)
