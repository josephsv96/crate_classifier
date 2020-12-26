from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid
import tensorflow as tf
import numpy as np
import os
import cv2
from rhs_utils import decode_netout, compute_overlap, compute_ap
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from rhs_preprocessing import YoloBatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#from yolo_backend import TinyYoloFeature
import keras
import sys
import matplotlib.pyplot as plt


class SpecialYOLO(object):
    def __init__(self, input_width,
                 input_height,
                 num_classes,
                 num_exposures,
                 class_weights):

        self.input_width = input_width
        self.input_height = input_height
        self.num_classes = num_classes
        self.num_exposures = num_exposures
        self.class_weights = class_weights

        ##########################
        # Make the model
        ##########################

        self.seen = 0

        # make the feature extractor layers
        input_image = Input(
            shape=(self.input_height, self.input_width, 3*num_exposures))
        num_layer = 0

        # stack 1, no reduction
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                   name='conv_' + str(num_layer), use_bias=False, kernel_initializer='random_normal')(input_image)
        x = BatchNormalization(name='norm_' + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)
        #x = MaxPooling2D(pool_size=(1, 1))(x)
        num_layer += 1
        print("stack 1")

        # stack 2, does not reduce extents but enlarges the area of influence for each convolution mask
        for i in range(0, 48):
            x = Conv2D(8, (3, 3), strides=(1, 1), padding='same', name='conv_' +
                       str(num_layer), use_bias=False, kernel_initializer='random_normal')(x)
            x = BatchNormalization(name='norm_' + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            #x = MaxPooling2D(pool_size=(1, 1))(x)
            num_layer += 1

        print("stack 2")
        # stack 3, does not reduce extents but enlarges the area of influence for each convolution mask
        for i in range(0, 1):
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_' +
                       str(num_layer), use_bias=False, kernel_initializer='random_normal')(x)
            x = BatchNormalization(name='norm_' + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            num_layer += 1

        print("stack 3")

        # stack 4, does not reduce extents5
        x = Conv2D(self.num_classes, (1, 1), strides=(1, 1), padding='same', name='conv_' +
                   str(num_layer), use_bias=False, kernel_initializer='random_normal')(x)
        x = BatchNormalization(name='norm_' + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1
        print("stack 4")

        # make the object detection layer, but its overwitten below!!
        output = Conv2D(self.num_classes,
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(x)

        print("x.shape=", x.shape.as_list())
        self.grid_h = x.shape.as_list()[1]
        self.grid_w = x.shape.as_list()[2]

        print("self.grid_h, self.grid_w=", self.grid_h, self.grid_w)

        output = Reshape((self.grid_h, self.grid_w, self.num_classes))(x)

        print("model_1 input shape=", input_image.shape)
        print("model_2 output shape=", output.shape)

        #self.model = Model([input_image, self.true_kpps], output)
        self.model = Model(inputs=input_image, outputs=output)

        # ----------------------------------------------------------------------------------------------
        # self.model.load_weights( "crates_new.h5" )
        # ----------------------------------------------------------------------------------------------

        # initialize the weights of the detection layer
        # layer = self.model.layers  #all layers
        #weights = layer.get_weights()

        #new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        #new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)

        #layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self.model.summary(positions=[.25, .60, .80, 1.])
        # tf.logging.set_verbosity(tf.logging.INFO)

    # y_true und y_pred sind die Daten zum gesamten Batch.
    def custom_loss(self, y_true, y_pred):
        # shape für y_pred (y_true sollte gleiches shape haben): <batch_size> <gridsize_x> <gridsize_y> <classes one-hot>
        # y_true, y_pred sind die Daten für einen ganzen batch

        # y_true = tf.Print( y_true, [1], message="***start*** \n", summarize=10000 )
        # y_true = tf.Print( y_true, [y_true], message="y_true= \n", summarize=10000 )
        # y_pred = tf.Print( y_pred, [y_pred], message="y_pred= \n", summarize=10000 )

        nb_cells = self.grid_w*self.grid_h
        pred_class = y_pred
        true_class = y_true
        #batch_size = tf.to_float( tf.shape( y_true )[0])
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

        # batch_size = tf.Print( batch_size, [batch_size], message="batch_size \n", summarize=500 )
        # tf.print(batch_size, output_stream=sys.stderr)

        #pred_class = tf.Print( pred_class, [pred_class], message="pred_class \n", summarize=500 )
        #true_class = tf.Print( true_class, [true_class], message="true_class \n", summarize=500 )

        # class_mask is made to reduce punishing of background class which is at index 0 in one-hot-class-vector
        # class_mask = tf.Variable( tf.ones( tf.shape( y_true )[-1]-1),dtype = tf.float32)

        class_mask = self.class_weights
        #class_mask = tf.ones( (tf.shape( y_true )[-1]-1))
        #class_mask = tf.concat( [[0.1], class_mask], 0 )
        class_mask = tf.tile(
            class_mask, [tf.reduce_prod(tf.shape(y_true)[:3])])
        class_mask = tf.reshape(class_mask, tf.shape(y_true))

        #class_mask = tf.Print( class_mask, [class_mask], message="class_mask \n", summarize=100000 )

        diff_sqr_class = tf.square(true_class-pred_class) * class_mask
        #diff_sqr_class = tf.Print( diff_sqr_class, [diff_sqr_class], message="diff_sgr_class \n", summarize=10000 )

        loss_class = tf.reduce_sum(diff_sqr_class) / \
            (nb_cells*self.num_classes*batch_size)

        #loss_class = tf.Print( loss_class, [loss_class], message="loss_class \n", summarize=100000 )

        # loss = tf.cond(tf.less(self.seen, self.warmup_batches+1),
        #               lambda: loss_class + 10 + loss_conf + 10,
        #               lambda: loss_class + loss_conf )

        loss = loss_class

        if self.debug:
            # loss = tf.Print(loss, [loss_kp0_xy], message='Loss Keyp0 \t', summarize=1000)
            # loss = tf.Print(loss, [loss_alpha], message='Loss alpha \t', summarize=1000)
            # loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            # loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            # loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            # loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            # loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)

            print_op = tf.print("total loss=", loss,
                                output_stream=sys.stderr, summarize=-1)
            with tf.control_dependencies([print_op]):
                closs = tf.add(loss, 0)

        self.seen += 1.0
        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)
        self.model.save(weight_path+"full")

        print("input layer name=")
        print([node.op.name for node in self.model.inputs])
        print("output layer name=")
        print([node.op.name for node in self.model.outputs])

        return self.model.output.shape[1:3]

    def normalize(self, image):
        return image / 255.0

    def train(self, train_imgs,     # the list of images to train the model
              valid_imgs,     # the list of images used to validate the model
              train_times,    # the number of time to repeat the training set, often used for small datasets
              valid_times,    # the number of times to repeat the validation set, often used for small datasets
              nb_epochs,      # number of epoches
              learning_rate,  # the learning rate
              batch_size,     # the size of the batch
              num_classes,
              warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
              saved_weights_name='crates.h5',
              debug=False):

        self.batch_size = batch_size

        self.debug = debug

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H': self.input_height,
            'IMAGE_W': self.input_width,
            'GRID_H': self.grid_h,
            'GRID_W': self.grid_w,
            'BATCH_SIZE': self.batch_size,
            'NUM_CLASSES': self.num_classes,
            'NUM_EXPOSURES': self.num_exposures
        }

        train_generator = YoloBatchGenerator(train_imgs,
                                             generator_config,
                                             norm=self.normalize)
        valid_generator = YoloBatchGenerator(valid_imgs,
                                             generator_config,
                                             norm=self.normalize,
                                             jitter=False)

        self.warmup_batches = warmup_epochs * \
            (train_times*len(train_generator) + valid_times*len(valid_generator))

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9,
                         beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   patience=500000,
                                   mode='min',
                                   verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='min',
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                                  histogram_freq=0,
                                  # write_batch_performance=True,
                                  write_graph=True,
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=len(
                                     train_generator) * train_times,
                                 epochs=warmup_epochs + nb_epochs,
                                 verbose=2 if debug else 1,
                                 validation_data=valid_generator,
                                 validation_steps=len(
                                     valid_generator) * valid_times,
                                 callbacks=[early_stop,
                                            checkpoint, tensorboard],
                                 workers=3,  # vormals 3
                                 max_queue_size=8)

        ############################################
        # Compute mAP on the validation set
        ############################################

        ##### test prediction ###########################
        print("test prediction start\n")
        image = cv2.imread("..\\images\\train\\t_img_001.bmp")
        # image = image[:,:,:] # all channels
        image = self.normalize(image)
        # x_batch, y_batch = train_generator.__getitem__( 0 )
        # for i in range( 0, x_batch.shape[0] ):
        #     image = x_batch[i]
        #     cv2.imwrite( "data\\aug_images\\augimg_"+str( i ) + ".bmp", image*255 )

        # image = x_batch[0]
        self.predict(image)
        print("test prediction end\n")
        ##### test prediction ende ######################
        # average_precisions = self.evaluate(valid_generator)

        # print evaluation
        # for label, average_precision in average_precisions.items():
        #     print(self.labels[label], '{:.4f}'.format(average_precision))
        # print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
        # for i in range( 512):
        #    cv2.imwrite( "data\\aug_images\\augimg_"+str( i ) + ".bmp", train_generator.proto_images[i] )

    def predict(self, images):
        #image_h, image_w, _ = image.shape
        #image = cv2.resize(image, (self.input_size, self.input_size))
        #image = self.normalize(image)

        # input_image = image #image[:,:,::-1] #flip rgb to bgr or vice versa
        # input_image = np.expand_dims(input_image, 0)  #why?
        # input_image = np.expand_dims(input_image, 0)

        netout = self.model.predict(images)[0]  # why?

        # print( "netout=", [netout] )  # print the netout

        # netout_decoded = decode_netout(netout, self.anchors, image_w, image_h, self.nb_class)

        return netout
