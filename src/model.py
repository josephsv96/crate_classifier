
from tensorflow.keras import Model
# Sequential

from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import add, Activation, Reshape, UpSampling2D
# Conv2DTranspose
from tensorflow.keras.layers import Conv2D, LeakyReLU, Softmax
# Softmax
from tensorflow.keras.layers import BatchNormalization
# average, concatenate
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.utils import plot_model


def cnn_def(input_layer, num_layer, num_classes):

    # stack 1, no reduction

    x = Conv2D(14, (3, 3), strides=(1, 1), padding='same',
               name='conv_' + str(num_layer), use_bias=False)(input_layer)
    x = BatchNormalization(name='norm_' + str(num_layer))(x)
    x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(1, 1))(x)

    num_layer += 1
    print("stack 1")

    # stack 2, does not reduce extends but enlarges the area of influence for
    # each convolution mask

    for i in range(0, 7):
        x = Conv2D(14, (3, 3), strides=(1, 1), padding='same',
                   name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = MaxPooling2D(pool_size=(1, 1))(x)

        num_layer += 1
    print("stack 2")

    # stack 3, does not reduce extends but enlarges the area of influence for
    # each convolution mask

    for i in range(0, 1):
        x = Conv2D(7, (1, 1), strides=(1, 1), padding='same',
                   name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1
    print("stack 3")

    # stack 4, does not reduce extends

    x = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='same',
               name='conv_'+str(num_layer), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(num_layer))(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = Softmax()(x)
    num_layer += 1
    print("stack 4")

    # Final Activation
    x = LeakyReLU(alpha=0.1)(x)

    # Output Detection layer
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='same',
               name='DetectionLayer', use_bias=False)(x)

    # output = Reshape((grid_h, grid_w, num_classes))(x)
    output = x

    return output


def cnn_def_2(input_layer, num_layer, num_classes):

    # stack 1, no reduction

    x = Conv2D(14, (3, 3), strides=(1, 1), padding='same',
               name='conv_' + str(num_layer), use_bias=False)(input_layer)
    x = BatchNormalization(name='norm_' + str(num_layer))(x)
    x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(1, 1))(x)

    num_layer += 1
    print("stack 1")

    # stack 2, does not reduce extends but enlarges the area of influence for
    # each convolution mask

    for i in range(0, 7):
        x = Conv2D(14, (3, 3), strides=(1, 1), padding='same',
                   name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = MaxPooling2D(pool_size=(1, 1))(x)

        num_layer += 1
    print("stack 2")

    # stack 3, does not reduce extends but enlarges the area of influence for
    # each convolution mask

    for i in range(0, 1):
        x = Conv2D(7, (1, 1), strides=(1, 1), padding='same',
                   name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1
    print("stack 3")

    stack_3_out = x

    # output_2
    x = Conv2D(7, (1, 1), strides=(1, 1), padding='same',
               name='conv_'+str(num_layer), use_bias=False)(stack_3_out)
    x = BatchNormalization(name='norm_' + str(num_layer))(x)
    num_layer += 1

    # Final Activation
    x = LeakyReLU(alpha=0.1)(x)
    # x = Softmax()(x)

    # Output Detection layer
    x = Conv2D(1, (1, 1), strides=(1, 1), padding='same',
               name='isCrate', use_bias=False)(x)

    output_layer_2 = x
    # END of output 2

    # output_1
    # stack 4, does not reduce extends

    x = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='same',
               name='conv_'+str(num_layer), use_bias=False)(stack_3_out)
    x = BatchNormalization(name='norm_' + str(num_layer))(x)
    num_layer += 1

    # Final Activation
    x = LeakyReLU(alpha=0.1)(x)

    # Output Detection layer
    x = Conv2D(num_classes, (1, 1), strides=(1, 1),
               padding='same', name='class', use_bias=False)(x)

    output_layer_1 = x
    # END of output 1

    print("stack 4")
    # END of Model defenition
    return output_layer_1, output_layer_2


def make_model(img_shape, num_classes, num_exposures=1, loss_func='CatCross'):
    height = img_shape[0]
    width = img_shape[1]
    input_layer = Input(shape=(height, width, 3*num_exposures))
    num_layer = 0

    output_layer = cnn_def(input_layer, num_layer, num_classes)

    # Model Defenition
    model = Model(inputs=input_layer, outputs=output_layer,
                  name='cnn_model')

    # Compiling the model
    if loss_func == 'CatCross':
        model.compile(optimizer='adam',
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    elif loss_func == 'BinCross':
        model.compile(optimizer='adam',
                      loss=BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    else:
        print('Error: Undefined Loss Function')

    return input_layer, model


def make_model_2(img_shape, num_classes, num_exposures=1, init_lr=1e-3, epochs=100):
    height = img_shape[0]
    width = img_shape[1]
    input_layer = Input(shape=(height, width, 3*num_exposures))
    num_layer = 0

    output_layer_1, output_layer_2 = cnn_def_2(
        input_layer, num_layer, num_classes)

    # Model Defenition
    model = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2],
                  name='cnn_model')

    # Compiling the model
    losses = {"isCrate": "binary_crossentropy",
              "class": "categorical_crossentropy"}

    lossWeights = {"isCrate": 1.0, "class": 1.0}
    # initialize the optimizer and compile the model

    opt = Adam(lr=init_lr, decay=init_lr / epochs)

    model.compile(optimizer=opt,
                  loss=losses,
                  loss_weights=lossWeights,
                  metrics=["accuracy"])

    return input_layer, model

# Funtion to change to reshape output layers


def model_reshape(model, input_tensor, depth, num_classes, num_layer=9, loss_func='CatCross'):
    """ To reshape the output layers of the output.
    Takes a trained model as input, adds layers to a branch at 'depth' level.
    The new_model is
    """

    # New Convolutional layers
    new_conv = Conv2D(num_classes, (1, 1),
                      strides=(1, 1),
                      padding='same',
                      name='conv_' + str(num_layer),
                      use_bias=False)(model.layers[depth].output)

    x = BatchNormalization(name='norm_' + str(num_layer))(new_conv)
    num_layer += 1

    # Final Activation
    x = LeakyReLU(alpha=0.1)(x)

    # Output Detection layer
    x = Conv2D(num_classes, (1, 1),
               strides=(1, 1),
               padding='same',
               name='DetectionLayer',
               use_bias=False)(x)

    output_layer = x

    # Defining the model
    new_model = Model(input_tensor, output_layer)

    if loss_func == 'CatCross':
        new_model.compile(optimizer='adam',
                          loss=CategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

    elif loss_func == 'BinCross':
        new_model.compile(optimizer='adam',
                          loss=BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])
    else:
        print('Error: Undefined Loss Function')

    return new_model


def model_freeze(model, depth):
    """To freeze the firt layers of a model
    depth - the number of last layer to remain trainable
    """
    layers_total = len(model.layers)
    layers_frozen = layers_total - depth
    # Freezing pretrained model
    for i in range(layers_frozen):
        model.layers[i].trainable = False

    # Training only newly trained model
    for i in range(layers_frozen, layers_total):
        model.layers[i].trainable = True

    return model
