import numpy as np
import collections.abc
import matplotlib.pyplot as plt


class Data:

    def __init__(self, data):
        self.data = data
        # self.out_dims is the shape of the output of this layer
        self.out_dims = data.shape

    def set_data(self, data):
        self.data = data
        # print(data.shape)

    def forward(self):
        return self.data


class conv2D:

    def __init__(self, in_layer, num_filters, filter_size, activation, T, bias):

        self.in_layer = in_layer
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.T = T
        self.bias = bias
        self.filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        self.laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.prewitt = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        self.square = np.array(
            [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]])
        self.padding = 1
        self.stride = 1

    def forward(self):

        self.in_array = self.in_layer.forward()
        # print("conv2D called")

        # Problem 1

        # kernel = np.flipud(np.fliplr(self.laplacian))

        # Problem 2

        kernel = np.flipud(np.fliplr(self.square))

        padding = self.padding
        strides = self.stride
        image = self.in_array
        # print("Shape of the image", image.shape)

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

        if self.activation == 'relu':
            output = np.maximum(output, 0)
            self.out_array = output
        else:
            self.out_array = output
        # plt.figure(1)
        # plt.imshow(output)
        # plt.figure(2)
        # plt.imshow(image)
        # plt.show()

        # print("Shape after convolution", output.shape)

        # Done so that it propagates

        return self.out_array


class pool2D:

    def __init__(self, in_layer, dim, type_pool):
        self.in_layer = in_layer
        self.dim = dim
        self.type_pool = type_pool
        self.stride = 2
        self.padding = 0
        pass

    def forward(self):
        self.in_array = self.in_layer.forward()
        # print(self.in_array)
        # print("pool2D called")

        image = self.in_array
        # plt.imshow(image)
        # plt.show()
        padding = self.padding
        strides = self.stride
        dim = self.in_array.shape
        xKernShape = dim[0]
        yKernShape = dim[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))
        # print(xKernShape)
        # print(yKernShape)

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
                            if self.type_pool == 'max':
                                output[x, y] = np.max(
                                    image[x: x + xKernShape, y: y + yKernShape])
                            elif self.type_pool == 'avg':
                                output[x, y] = np.mean(
                                    image[x: x + xKernShape, y: y + yKernShape])

                    except:
                        break

        # plt.figure(1)
        # plt.imshow(output)
        # plt.figure(2)
        # plt.imshow(image)
        # plt.show()

        # print("Shape after pooling", output.shape)

        self.out_array = output

        return self.out_array


class full2D:

    def __init__(self, in_layer, bias, T):
        self.in_layer = in_layer
        self.T = T
        self.bias = bias
        self.bias_P1 = 3500
        self.bias_P2 = 40000
        self.bias_P3 = 4000
        pass

    def forward(self):
        self.in_array = self.in_layer.forward()
        # print(self.in_array)
        self.temp_W = np.ones(self.in_array.shape)
        self.out_array = self.in_array.ravel().dot(self.temp_W.ravel())

        # Task 1
        # self.out_array = self.out_array - self.bias_P1

        # Task 2
        # self.out_array = self.out_array - self.bias_P2
        # print("Pre Sigmoid", self.out_array)

        # Task 3
        self.out_array = self.out_array - self.bias_P3
        # print("Raw pre Sigmoid", self.out_array)

        if self.out_array >= 0:
            self.out_array = 1 / (1. + np.exp(-self.out_array))
        else:
            self.out_array = np.exp(self.out_array) / \
                (1. + np.exp(self.out_array))
        # print("After Sigmoid", self.out_array)
        # print("full2D called")

        # print("Shape at the output", self.in_array.shape)

        return self.out_array


class Relu:
    """Given an input matrix X, with one feature vector per row,
    this layer computes maximum(X,0), where the maximum operator is coordinate-wise."""

    def __init__(self, in_layer):
        self.in_layer = in_layer
        self.in_dims = in_layer.out_dims

        # TODO: Set out_dims to the shape of the output of this relu layer as a numpy array e.g. self.out_dims = np.array([...])
        self.out_dims = self.in_dims

    def forward(self):
        self.in_array = self.in_layer.forward()

        # TODO: Compute the result of Relu function, and store it as self.out_array
        self.out_array = np.maximum(self.in_array, 0)
        return self.out_array

    pass


class Sigmoid:

    # FIX NUMERICAL STABILITY

    def __init__(self, in_layer):
        self.in_layer = in_layer

    def forward(self):
        self.in_array = self.in_layer.forward()

        # TODO: Compute the result of sigmoid function, and store it as self.out_array. Be careful! Don't exponentiate an arbitrary positive number as it may overflow.
        self.out_array = 1 / (1 + np.exp(-self.in_array))

        return self.out_array
