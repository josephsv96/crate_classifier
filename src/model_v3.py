# import the necessary packages

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class CrateNet:
    def __init__(self, grid_h, grid_w, num_exp, num_classes, init_lr, epochs):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_exp = num_exp
        self.num_classes = num_classes

        self.init_lr = init_lr
        self.epochs = epochs

    @staticmethod
    def denife_cnn(height, width, num_exposures, num_classes, depth=3):
        input_layer = Input(shape=(height, width, depth*num_exposures))
        num_layer = 0

        # stack 1, no reduction
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                   name='conv_' + str(num_layer), use_bias=False)(input_layer)
        x = BatchNormalization(name='norm_' + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = MaxPooling2D(pool_size=(1, 1))(x)

        num_layer += 1
        print("stack 1")

        # stack 2, does not reduce extends but enlarges the area of influence
        # for each convolution mask
        for i in range(0, 48):
            x = Conv2D(8, (3, 3), strides=(1, 1), padding='same',
                       name='conv_' + str(num_layer), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            # x = MaxPooling2D(pool_size=(1, 1))(x)

            num_layer += 1
        print("stack 2")

        # stack 3, does not reduce extends but enlarges the area of influence
        # for each convolution mask
        for i in range(0, 1):
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='same',
                       name='conv_' + str(num_layer), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            num_layer += 1
        print("stack 3")

        # stack 4, does not reduce extends
        x = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='same',
                   name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)

        num_layer += 1
        print("stack 4")

        # Final Activation
        # x = LeakyReLU(alpha=0.1)(x)

        # Output Detection layer
        x = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='same',
                   name='DetectionLayer', use_bias=False)(x)

        output_layer = Reshape((height, width, num_classes))(x)

        return input_layer, output_layer

    @staticmethod
    def build(grid_h, grid_w, num_exp, num_classes, init_lr, epochs):
        """Build model with CategoricalCrossentropy loss
        """
        input_l, output_l = CrateNet.denife_cnn(height=grid_h,
                                                width=grid_w,
                                                num_exposures=num_exp,
                                                num_classes=num_classes)
        # Model Defenition
        model = Model(inputs=input_l, outputs=output_l,
                      name="cnn_model_" + str(num_exp) + "_exp")

        opt = Adam(lr=init_lr,
                   decay=init_lr / (epochs * 0.5))

        model.compile(loss=CategoricalCrossentropy(from_logits=True),
                      optimizer=opt,
                      metrics=["accuracy"])
        return model
