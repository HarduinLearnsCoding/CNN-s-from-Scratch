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


def perform_classification(task, images):

    dataset_size = (200, 128, 128)

    images_Task1, images_Task2, images_Task3 = images

    # print(images_Task1.shape)
    # print(images_Task2.shape)
    # print(images_Task3.shape)

    images_Task1_Label = np.zeros((images_Task1.shape[0], 1))
    images_Task2_Label = np.zeros((images_Task2.shape[0], 1))
    images_Task3_Label = np.zeros((images_Task3.shape[0], 1))

    # Labeling

    images_Task1_Label[:100, :] = 1
    images_Task1_Label[100:, :] = 0

    images_Task2_Label[:100, :] = 0
    images_Task2_Label[100:, :] = 1

    images_Task3_Label[:100, :] = 1
    images_Task3_Label[100:, :] = 0

    # print(images_Task1_Label.shape)
    # print(images_Task2_Label.shape)
    # print(images_Task3_Label.shape)

    model = create_model(dataset_size)

    if task == '1':

        images_Task1 = images_Task1[:, np.newaxis, :, :]

        model.layers_list[0].set_data(images_Task1)

        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        laplacian=laplacian[np.newaxis,:,:]

        bias = 0

        filter_size = laplacian.shape

        num_channels = 1

        activation = 'relu'

        dim = [3, 3]

        type_pool = 'avg'

        bias_FC = 3500

        T_FC = None   # Automatically get shape from previous layer

        model = add_conv_layer(
            model, num_channels, filter_size, activation, laplacian, bias)
        model = add_pooling_layer(model, dim, type_pool)
        model = add_FC_sigmoid_layer(model, bias_FC, T_FC)

        predict_label_Task1 = np.zeros((images_Task1.shape[0], 1))

        for i in range(images_Task1.shape[0]):
            model.layers_list[0].set_data(images_Task1[i, :, :])
            prediction = model.layers_list[-1].forward()
            print("Image: ", i)
            predict_label_Task1[i] = prediction

        # print(predict_label_Task1)

        count = 0

        for i in range(images_Task1.shape[0]):
            if predict_label_Task1[i] == images_Task1_Label[i]:
                count += 1

        print("Accuracy ", count / images_Task1.shape[0])

    elif task == '2':

        prewitt = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

        prewitt = prewitt[np.newaxis,:,:]

        bias = 0

        filter_size = prewitt.shape

        num_channels = 1

        activation = 'relu'

        dim = [3, 3]

        type_pool = 'avg'

        bias_FC = 40500

        T_FC = None   # Automatically get shape from previous layer

        images_Task2 = images_Task2[:, np.newaxis, :, :]

        model.layers_list[0].set_data(images_Task2)

        model = add_conv_layer(
            model, num_channels, filter_size, activation, prewitt, bias)
        model = add_pooling_layer(model, dim, type_pool)
        model = add_FC_sigmoid_layer(model, bias_FC, T_FC)

        predict_label_Task2 = np.zeros((images_Task2.shape[0], 1))

        for i in range(images_Task2.shape[0]):
            model.layers_list[0].set_data(images_Task2[i, :,:, :])
            prediction = model.layers_list[-1].forward()
            print("Image: ", i)
            predict_label_Task2[i] = prediction

        # print(predict_label_Task1)

        count = 0

        for i in range(images_Task2.shape[0]):
            if predict_label_Task2[i] == images_Task2_Label[i]:
                count += 1

        print("Accuracy ", count / images_Task2.shape[0])

    elif task == '3b':

        kernel_feed=input("Which kernel size? (1 : 3x3,  2 : 5x5) \n")


        images_Task3 = images_Task3[:, np.newaxis, :, :]

        # print(images_Task3.shape)

        model.layers_list[0].set_data(images_Task3)

        # square = np.array(
        #     [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]])

        square3x3 = np.array([[[1, 1, 1], [1, 0, 0], [1, 0, 0]], [[
                             1, 1, 1], [0, 0, 1], [0, 0, 1]], [[1, 0, 0], [1, 0, 0], [1, 1, 1]], [[0, 0, 1], [0, 0, 1], [1, 1, 1]]])

        squarepointy=np.array([[[1,0,1],[0,0,0],[1,0,1]],[[1,0,1],[0,0,0],[1,0,1]],[[1,0,1],[0,0,0],[1,0,1]],[[1,0,1],[0,0,0],[1,0,1]]])

        square5x5= np.array([[[1, 1, 1,1,1], [1, 0, 0,0,0], [1, 0, 0,0,0],[1, 0, 0,0,0],[1, 0, 0,0,0]], [[
                             1, 1, 1,1,1], [0, 0, 0,0,1], [0, 0, 0,0,1],[0, 0, 0,0,1],[0, 0, 0,0,1]], [[
                             0, 0, 0,0,1], [0, 0, 0,0,1], [0, 0, 0,0,1],[0, 0, 0,0,1],[1, 1, 1,1,1]], [[
                             1, 0, 0,0,0], [1, 0, 0,0,0], [1, 0, 0,0,0],[1, 0, 0,0,0],[1, 1, 1,1,1]]])

        # impulse=np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]]])

        # print("Pointy Shape:", squarepointy.shape)
        # print("Kernel Shape: ",square3x3.shape)

        bias = 0

        filter_size = square3x3.shape

        num_channels = 1

        dim = [2,2]

        bias_FC = 0

        type_pool = 'max'

        activation = 'relu'

        dimglobal=None

        T_FC = None   # Automatically get shape from previous layer

        if kernel_feed=='1':
            kernel=square3x3
            bias_FC=13000
        elif kernel_feed=='2':
            kernel=square5x5
            bias_FC=17500

        model = add_conv_layer(
            model, num_channels, filter_size, activation, kernel, bias)
        model = add_pooling_layer(model, dim, type_pool)

        num_channels_2=4

        filter_size_2= squarepointy.shape

        model= add_conv_layer(
            model, num_channels_2, filter_size_2, activation, squarepointy, bias)
        model=add_pooling_layer(model, dimglobal, type_pool)
        model = add_FC_sigmoid_layer(model, bias_FC, T_FC)

        predict_label_Task3 = np.zeros((images_Task3.shape[0], 1))

        for i in range(images_Task3.shape[0]):
        # for i in range(1):
            model.layers_list[0].set_data(images_Task3[i, :, :, :])
            prediction = model.layers_list[-1].forward()
            print("Image: ", i)
            predict_label_Task3[i] = prediction

        # print(predict_label_Task1)

        count = 0

        for i in range(images_Task3.shape[0]):
            if predict_label_Task3[i] == images_Task3_Label[i]:
                count += 1

        print("Accuracy ", count / images_Task3.shape[0])

    elif task=='3':

        images_Task3 = images_Task3[:, np.newaxis, :, :]

        # print(images_Task3.shape)

        model.layers_list[0].set_data(images_Task3)

        square = np.array(
            [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]])

        square=square[np.newaxis,:,:]

        bias = 0

        filter_size = square.shape

        num_channels = 1

        dim = [2,2]

        bias_FC = 4000

        type_pool = 'max'

        activation = 'none'

        dimglobal=None

        T_FC = None   # Automatically get shape from previous layer

        model = add_conv_layer(
            model, num_channels, filter_size, activation, square, bias)
        model=add_pooling_layer(model, dimglobal, type_pool)
        model = add_FC_sigmoid_layer(model, bias_FC, T_FC)

        predict_label_Task3 = np.zeros((images_Task3.shape[0], 1))

        for i in range(images_Task3.shape[0]):
        # for i in range(1):
            model.layers_list[0].set_data(images_Task3[i, :, :, :])
            prediction = model.layers_list[-1].forward()
            print("Image: ", i)
            predict_label_Task3[i] = prediction

        # print(predict_label_Task1)

        count = 0

        for i in range(images_Task3.shape[0]):
            if predict_label_Task3[i] == images_Task3_Label[i]:
                count += 1

        print("Accuracy ", count / images_Task3.shape[0])

    pass


class Model():

    def __init__(self, input_size):

        self.input_size = input_size
        self.layers_list = []


def main():

    # Reading in Images

    # dataset_size = (200, 128, 128)

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

    images_Task1 = np.concatenate(
        (images_Task1_Class0, images_Task1_Class1))
    images_Task2 = np.concatenate(
        (images_Task2_Class0, images_Task2_Class1))
    images_Task3 = np.concatenate(
        (images_Task3_Class0, images_Task3_Class1))

    images = [images_Task1, images_Task2, images_Task3]

    task = input(
        "Which task do you want to perform ? (1 for Task 1 (Implemented with EC) , 2 for Task 2, 3 for Task 3, 3b for Task 3 with EC) \nEC stands for Extra Credit \n ")

    perform_classification(task, images)

    pass


if __name__ == "__main__":
    main()
    pass
