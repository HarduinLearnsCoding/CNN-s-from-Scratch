import imageio as iio
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import layers


def create_model(input_size):
    model = Model(input_size)
    default_data = np.zeros((input_size))
    data_layer = layers.Data(default_data)
    model.layers_list.append(data_layer)
    return model


def add_conv_layer(model, num_channels, filter_size, activation, T, b):
    # conv_layer = layers.conv2D(num_channels, filter_size, activation, T, b)
    # print(model.layers_list[-1])
    conv_layer = layers.conv2D(
        model.layers_list[-1], num_channels, filter_size, activation, T, b)
    model.layers_list.append(conv_layer)
    return model


def add_pooling_layer(model, dim, type):
    # pool_layer = layers.pool2D(dim, type)
    # print(model.layers_list[-1])
    pool_layer = layers.pool2D(model.layers_list[-1], dim, type)
    model.layers_list.append(pool_layer)
    return model


def add_FC_sigmoid_layer(model, b, T):
    # fully_layer = layers.full2D(b, T)
    # print(model.layers_list[-1])
    fully_layer = layers.full2D(model.layers_list[-1], b, T)
    model.layers_list.append(fully_layer)
    return model


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


def perform_classification(task, images):
    pass


def maxpool2D(image, kernel_size, padding=0, strides=1):

    xKernShape = kernel_size
    yKernShape = kernel_size
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
                            np.max(imagePadded[x: x + xKernShape, y: y + yKernShape]))
                except:
                    break


class Model():

    def __init__(self, input_size):

        self.input_size = input_size
        self.layers_list = []


def main():

    # Reading in Images

    dataset_size = (128, 128, 200)

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

    images_Task1 = np.zeros((128, 128, 200))
    images_Task2 = np.zeros((128, 128, 200))
    images_Task3 = np.zeros((128, 128, 200))

    images_Task1_preshaping = np.concatenate(
        (images_Task1_Class0, images_Task1_Class1))
    images_Task2_preshaping = np.concatenate(
        (images_Task2_Class0, images_Task2_Class1))
    images_Task3_preshaping = np.concatenate(
        (images_Task3_Class0, images_Task3_Class1))

    for i in range(images_Task1.shape[0]):
        images_Task1[:, :, i] = images_Task1_preshaping[i, :, :]

    for i in range(images_Task1.shape[0]):
        images_Task2[:, :, i] = images_Task2_preshaping[i, :, :]

    for i in range(images_Task1.shape[0]):
        images_Task3[:, :, i] = images_Task3_preshaping[i, :, :]

    # print(images_Task1.shape)
    # print(images_Task2.shape)
    # print(images_Task3.shape)

    model = create_model(dataset_size)

    # print(type(model))

    # Setting Data

    model.layers_list[0].set_data(images_Task1)

    model = add_conv_layer(model, 1, 1, 1, 1, 1)
    model = add_pooling_layer(model, 1, 1)
    model = add_FC_sigmoid_layer(model, 1, 1)

    # DEBUGGING ----------------------------------------------------------

    # print(model.layers_list)
    print(model.layers_list[-1].forward())
    # print(model.layers_list[-1].forward().shape)
    # print(model.layers_list[0].forward())

    # print(model.layers_list[0].forward())

    # print(images_Task1.shape)
    # print(images_Task2.shape)
    # print(images_Task3.shape)

    # print((images_Task1_preshaping[0, :, :] == images_Task1[:, :, 0]).all())
    # print((images_Task1_preshaping[1, :, :] == images_Task1[:, :, 1]).all())
    # print((images_Task1_preshaping[0, :, :] ==
    #        images_Task1_Class0[0, :, :]).all())

    # Testing convolve2D

    # kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    # output = convolve2D(images_Task1_Class0[1, :, :], kernel, padding=0)
    # plt.figure(1)
    # plt.imshow(images_Task1_Class0[1, :, :], 'gray')
    # plt.figure(2)
    # plt.imshow(output, 'gray')
    # plt.show()

    # model.layers_list[0].set_data(images_Task1)

    # print(images_Task1_Class0.shape)

    pass


if __name__ == "__main__":
    main()
    pass
