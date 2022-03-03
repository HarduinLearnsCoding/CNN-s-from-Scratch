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
        self.padding = 0
        self.stride = 1

    def forward(self):

        self.in_array = self.in_layer.forward()
        print("conv2D called")

        kernel = np.flipud(np.fliplr(self.filter))
        padding = self.padding
        strides = self.stride
        image = self.in_array[:, :, 0]
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
            self.out_array = output[:, :, None]
        else
            self.out_array = output[:, :, None]
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
        self.stride = 1
        self.padding = 0
        pass

    def forward(self):
        self.in_array = self.in_layer.forward()
        print("pool2D called")

        image = self.in_array[:, :, 0]
        # plt.imshow(image)
        # plt.show()
        padding = self.padding
        strides = self.stride
        dim = [3, 3]
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
                            output[x, y] = np.max(
                                image[x: x + xKernShape, y: y + yKernShape]).sum()
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
        pass

    def forward(self):
        self.in_array = self.in_layer.forward()

        print("full2D called")

        # print("Shape at the output", self.in_array.shape)

        return self.in_array


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


# DO NOT CHANGE ANY CODE IN THIS CLASS!


# class ModuleList(collections.abc.MutableSequence):
#     def __init__(self, *args):
#         self.list = list()
#         self.list.extend(list(args))
#         pass

#     def __getitem__(self, i):
#         return self.list[i]

#     def __setitem__(self, i, v):
#         self.list[i] = v

#     def __delitem__(self, i):
#         del self.list[i]
#         pass

#     def __len__(self):
#         return len(self.list)

#     def insert(self, i, v):
#         self.list.insert(i, v)
#         pass

#     def get_modules_with_parameters(self):
#         modules_with_parameters = []
#         for mod in self.list:
#             if is_modules_with_parameters(mod):
#                 modules_with_parameters.append(mod)
#                 pass
#             pass
#         return modules_with_parameters
#     pass


# class BaseNetwork:
#     def __init__(self):
#         super().__setattr__("initialized", True)
#         super().__setattr__("modules_with_parameters", [])
#         super().__setattr__("output_layer", None)

#     def set_output_layer(self, layer):
#         super().__setattr__("output_layer", layer)
#         pass

#     def get_output_layer(self):
#         return self.output_layer

#     def __setattr__(self, name, value):
#         if not hasattr(self, "initialized") or (not self.initialized):
#             raise RuntimeError(
#                 "You must call super().__init__() before assigning any layer in __init__().")
#         if is_modules_with_parameters(value) or isinstance(value, ModuleList):
#             self.modules_with_parameters.append(value)
#             pass

#         super().__setattr__(name, value)
#         pass

#     def get_modules_with_parameters(self):
#         modules_with_parameters_list = []
#         for mod in self.modules_with_parameters:
#             if isinstance(mod, ModuleList):

#                 modules_with_parameters_list.extend(
#                     mod.get_modules_with_parameters())
#                 pass
#             else:

#                 modules_with_parameters_list.append(mod)
#                 pass
#             pass
#         return modules_with_parameters_list

#     def forward(self):
#         return self.output_layer.forward()

#     def backward(self, input_grad):
#         self.output_layer.backward(input_grad)
#         pass

#     def state_dict(self):
#         all_params = []
#         for m in self.get_modules_with_parameters():
#             all_params.append(m.W)
#             pass
#         return all_params

#     def load_state_dict(self, state_dict):
#         assert len(state_dict) == len(self.get_modules_with_parameters())
#         for m, lw in zip(self.get_modules_with_parameters(), state_dict):
#             m.W = lw
#             pass
#         pass
#     pass
