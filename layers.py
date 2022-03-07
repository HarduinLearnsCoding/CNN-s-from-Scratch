import numpy as np
import collections.abc
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=10000000000)


class Data:

    def __init__(self, data):
        self.data = data
        self.out_dims = data.shape

    def set_data(self, data):
        self.data = data

    def forward(self):
        return self.data


class conv2D:

    def __init__(self, in_layer, num_channels, filter_size, activation, T, bias):

        self.in_layer = in_layer
        self.num_channels = num_channels
        self.filter_size = filter_size
        self.activation = activation
        self.T = T
        self.bias = bias
        self.padding = 1
        self.stride = 1

    def forward(self):

        # print("Conv2D Called")
        np.set_printoptions(threshold=np.inf, linewidth=10000000000)
        self.in_array = self.in_layer.forward()

        padding = self.padding
        strides = self.stride
        image = self.in_array
        # for i in range(self.in_array.shape[0]):
        #     print("Image Channel {}:".format(i), image[i, :, :40])

        xImgShape = image.shape[1]
        yImgShape = image.shape[2]
        self.num_filters = self.T.shape[0]

        if self.num_channels == self.num_filters:

            appendout = []

            for i in range(self.num_filters):

                kernel = self.T[i, :, :]
                image = self.in_array[i, :, :]
                kernel = np.flipud(np.fliplr(kernel))
                xKernShape = kernel.shape[0]
                yKernShape = kernel.shape[1]

                xOutput = int(
                    ((xImgShape - xKernShape + 2 * padding) / strides) + 1)
                yOutput = int(
                    ((yImgShape - yKernShape + 2 * padding) / strides) + 1)
                output = np.zeros((xOutput, yOutput))

                # print("Images Shape:", image.shape)

                if padding != 0:
                    imagePadded = np.zeros(
                        (image.shape[0] + padding * 2, image.shape[1] + padding * 2))
                    imagePadded[int(padding):int(-1 * padding),
                                int(padding):int(-1 * padding)] = image

                else:
                    imagePadded = image

                # Iterate through image
                for y in range(image.shape[1]):
                    # Exit Convolution
                    if y > image.shape[1] - yKernShape:
                        break

                    if y % strides == 0:
                        for x in range(image.shape[0]):

                            if x > image.shape[0] - xKernShape:
                                break
                            try:

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

                appendout.append(output)

            self.out_array = sum(appendout)

            self.out_array = self.out_array[np.newaxis, :, :]

            # print("IF CONV2D \n", np.maximum(
            #     self.out_array[0, :, :40] - 1274, 0))

            # print("if", self.out_array.shape)

        else:

            appendout = []

            for i in range(self.num_filters):

                # print(self.T.shape)
                image = self.in_array[0, :, :]
                kernel = self.T[i, :, :]
                kernel = np.flipud(np.fliplr(kernel))

                xKernShape = kernel.shape[0]
                yKernShape = kernel.shape[1]

                xOutput = int(
                    ((xImgShape - xKernShape + 2 * padding) / strides) + 1)
                yOutput = int(
                    ((yImgShape - yKernShape + 2 * padding) / strides) + 1)
                output = np.zeros((xOutput, yOutput))

                if padding != 0:
                    imagePadded = np.zeros(
                        (image.shape[0] + padding * 2, image.shape[1] + padding * 2))
                    imagePadded[int(padding):int(-1 * padding),
                                int(padding):int(-1 * padding)] = image

                else:
                    imagePadded = image

                # Iterate through image
                for y in range(image.shape[1]):
                    # Exit Convolution
                    if y > image.shape[1] - yKernShape:
                        break

                    if y % strides == 0:
                        for x in range(image.shape[0]):

                            if x > image.shape[0] - xKernShape:
                                break
                            try:

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

                appendout.append(output)

            self.out_array = np.stack(appendout)

            # for i in range(self.out_array.shape[0]):
            #     print("Channel {} \n".format(i), np.maximum(
            #         self.out_array[i, :, :40] - 1274, 0))
            # print("Channel {} \n".format(i),
            #       self.out_array[i, :, :40])
            # print("Else", self.out_array.shape)

        # print("Conv2D Out Shape", self.out_array.shape)

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

        # print("Pool2D Called")

        self.in_array = self.in_layer.forward()

        imagenew = self.in_array
        padding = self.padding
        strides = self.stride
        self.num_filters = self.in_array.shape[0]

        if self.dim == None:
            dim = self.in_array.shape
        else:
            dim = self.dim

        appendout = []

        # print("image Shape: ", imagenew.shape)

        for i in range(self.num_filters):

            image = imagenew[i, :, :]
            if self.dim == None:
                xKernShape = dim[1]
                yKernShape = dim[2]
            else:
                xKernShape = dim[0]
                yKernShape = dim[1]
            xImgShape = image.shape[0]
            yImgShape = image.shape[1]
            xOutput = int(
                ((xImgShape - xKernShape + 2 * padding) / strides) + 1)
            yOutput = int(
                ((yImgShape - yKernShape + 2 * padding) / strides) + 1)
            output = np.zeros((xOutput, yOutput))

            for y in range(image.shape[1]):

                if y > image.shape[1] - yKernShape:
                    break

                if y % strides == 0:
                    for x in range(image.shape[0]):

                        if x > image.shape[0] - xKernShape:
                            break
                        try:

                            if x % strides == 0:
                                if self.type_pool == 'max':
                                    output[x, y] = np.max(
                                        image[x: x + xKernShape, y: y + yKernShape])
                                elif self.type_pool == 'avg':
                                    output[x, y] = np.mean(
                                        image[x: x + xKernShape, y: y + yKernShape])

                        except:
                            break

            appendout.append(output)

        self.out_array = np.stack(appendout)

        # self.out_array = output

        # print("Pool2D Out Shape", self.out_array.shape)

        return self.out_array


class full2D:

    def __init__(self, in_layer, bias, T):
        self.in_layer = in_layer
        self.T = T
        self.bias = bias
        pass

    def forward(self):
        self.in_array = self.in_layer.forward()
        if self.T == None:
            self.W = np.ones(self.in_array.shape)

        self.out_array = self.in_array.ravel().dot(self.W.ravel())

        self.out_array = self.out_array - self.bias
        print("Raw pre Sigmoid", self.out_array)

        if self.out_array >= 0:
            self.out_array = 1 / (1. + np.exp(-self.out_array))
        else:
            self.out_array = np.exp(self.out_array) / \
                (1. + np.exp(self.out_array))
        # print("After Sigmoid", self.out_array)

        return self.out_array
