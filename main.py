import imageio as iio
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import layers


def create_model(input_size):
    pass


def add_conv_layer(model, num_channels, filter_size, activation, T, b):
    pass


def add_pooling_layer(model, dim, type):
    pass


def add_FC_sigmoid_layer(model, b, T):
    pass


def convolve2D(image, kernel, padding=0, strides=1):

    kernel = np.flipud(np.fliplr(kernel))
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    if padding != 0:
        imagePadded = np.zeros(
            (image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding),
                    int(padding):int(-1 * padding)] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (
                            kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


class Network(layers.BaseNetwork):
    # TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist

    def __init__(self, data_layer, hidden_units, hidden_layers):

        # you should always call __init__ first

        super().__init__()

        # TODO: define our network architecture here

        # self.layer_2_linear = layers.Linear(data_layer, hidden_units)
        # self.layer_3_bias = layers.Bias(self.layer_2_linear)
        # self.layer_4_relu = layers.Relu(self.layer_3_bias)
        # self.layer_5_linear = layers.Linear(self.layer_4_relu, 1)
        # self.layer_6_bias = layers.Bias(self.layer_5_linear)

        self.MY_MODULE_LIST = layers.ModuleList()
        self.MY_MODULE_LIST.append(data_layer)
        for i in range(hidden_layers):
            self.MY_MODULE_LIST.append(layers.Linear(
                self.MY_MODULE_LIST[-1], hidden_units[i]))
            self.MY_MODULE_LIST.append(layers.Bias(self.MY_MODULE_LIST[-1]))
            self.MY_MODULE_LIST.append(layers.Relu(self.MY_MODULE_LIST[-1]))

        self.MY_MODULE_LIST.append(layers.Linear(self.MY_MODULE_LIST[-1], 1))
        self.MY_MODULE_LIST.append(layers.Bias(self.MY_MODULE_LIST[-1]))
        # TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)

        self.set_output_layer(self.MY_MODULE_LIST[-1])


def main():

    # Reading in Images

    images_Task1_Class0 = list()
    for file in Path("Images/PS2_Images_Task_1_Class_0").iterdir():
        im = iio.imread(file)
        images_Task1_Class0.append(im)

    images_Task1_Class1 = list()
    for file in Path("Images/PS2_Images_Task_1_Class_1").iterdir():
        im = iio.imread(file)
        images_Task1_Class1.append(im)

    images_Task2_Class0 = list()
    for file in Path("Images/PS2_Images_Task_2_Class_0").iterdir():
        im = iio.imread(file)
        images_Task2_Class0.append(im)

    images_Task2_Class1 = list()
    for file in Path("Images/PS2_Images_Task_2_Class_1").iterdir():
        im = iio.imread(file)
        images_Task2_Class1.append(im)

    images_Task3_Class0 = list()
    for file in Path("Images/PS2_Images_Task_3_Class_0").iterdir():
        im = iio.imread(file)
        images_Task3_Class0.append(im)

    images_Task3_Class1 = list()
    for file in Path("Images/PS2_Images_Task_3_Class_1").iterdir():
        im = iio.imread(file)
        images_Task3_Class1.append(im)

    images_Task1_Class0 = np.array(images_Task1_Class0)
    images_Task1_Class1 = np.array(images_Task1_Class1)
    images_Task2_Class0 = np.array(images_Task2_Class0)
    images_Task2_Class1 = np.array(images_Task2_Class1)
    images_Task3_Class0 = np.array(images_Task3_Class0)
    images_Task3_Class1 = np.array(images_Task3_Class1)

    # Testing convolve2D

    # kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    # output = convolve2D(images_Task1_Class0[1, :, :], kernel, padding=0)
    # plt.figure(1)
    # plt.imshow(images_Task1_Class0[1, :, :])
    # plt.figure(2)
    # plt.imshow(output)
    # plt.show()

    # print(images_Task1_Class0.shape)

    pass


if __name__ == "__main__":
    main()
    pass
