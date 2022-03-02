import numpy as np
import collections.abc


class Data:
    """Stores an input array of training data, and hands it to the next layer."""

    def __init__(self, data):
        self.data = data
        # self.out_dims is the shape of the output of this layer
        self.out_dims = data.shape

    def set_data(self, data):
        self.data = data

    def forward(self):
        return self.data


class Conv:
    """Given an input matrix X, with one feature vector per row,
    this layer computes XW, where W is a linear operator."""

    def __init__(self, in_layer, num_out_features):
        assert len(
            in_layer.out_dims) == 2, "Input layer must contain a list of 1D linear feature data."

        self.in_layer = in_layer
        num_data, num_in_features = in_layer.out_dims

        # TODO: Set out_dims to the shape of the output of this linear layer as a numpy array e.g. self.out_dims = np.array([x, y])
        self.out_dims = np.array([num_data, num_out_features])

        # TODO: Declare the weight matrix. Be careful how you initialize the matrix.

        self.W = np.random.randn(
            num_in_features, num_out_features) / np.sqrt(num_in_features)

    def forward(self):
        """This function computes XW"""
        self.in_array = self.in_layer.forward()

        # TODO: Compute the result of linear layer with weight W, and store it as self.out_array
        self.out_array = self.in_array @ self.W

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


class Bias:
    """Given an input matrix X, add a trainable constant to each entry."""

    def __init__(self, in_layer):
        self.in_layer = in_layer
        num_data, num_in_features = in_layer.out_dims
        # TODO: Set out_dims to the shape of the output of this linear layer as a numpy array.
        self.out_dims = np.array([num_data, num_in_features])
        # TODO: Declare the weight matrix. Be careful how you initialize the matrix.
        self.W = np.random.randn(1, num_in_features)

    def forward(self):
        self.in_array = self.in_layer.forward()
        # TODO: Compute the result of Bias layer, and store it as self.out_array
        self.out_array = self.in_array + self.W
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


def is_modules_with_parameters(value):
    return isinstance(value, Linear) or isinstance(value, Bias)

# DO NOT CHANGE ANY CODE IN THIS CLASS!


class ModuleList(collections.abc.MutableSequence):
    def __init__(self, *args):
        self.list = list()
        self.list.extend(list(args))
        pass

    def __getitem__(self, i):
        return self.list[i]

    def __setitem__(self, i, v):
        self.list[i] = v

    def __delitem__(self, i):
        del self.list[i]
        pass

    def __len__(self):
        return len(self.list)

    def insert(self, i, v):
        self.list.insert(i, v)
        pass

    def get_modules_with_parameters(self):
        modules_with_parameters = []
        for mod in self.list:
            if is_modules_with_parameters(mod):
                modules_with_parameters.append(mod)
                pass
            pass
        return modules_with_parameters
    pass


class BaseNetwork:
    def __init__(self):
        super().__setattr__("initialized", True)
        super().__setattr__("modules_with_parameters", [])
        super().__setattr__("output_layer", None)

    def set_output_layer(self, layer):
        super().__setattr__("output_layer", layer)
        pass

    def get_output_layer(self):
        return self.output_layer

    def __setattr__(self, name, value):
        if not hasattr(self, "initialized") or (not self.initialized):
            raise RuntimeError(
                "You must call super().__init__() before assigning any layer in __init__().")
        if is_modules_with_parameters(value) or isinstance(value, ModuleList):
            self.modules_with_parameters.append(value)
            pass

        super().__setattr__(name, value)
        pass

    def get_modules_with_parameters(self):
        modules_with_parameters_list = []
        for mod in self.modules_with_parameters:
            if isinstance(mod, ModuleList):

                modules_with_parameters_list.extend(
                    mod.get_modules_with_parameters())
                pass
            else:

                modules_with_parameters_list.append(mod)
                pass
            pass
        return modules_with_parameters_list

    def forward(self):
        return self.output_layer.forward()

    def backward(self, input_grad):
        self.output_layer.backward(input_grad)
        pass

    def state_dict(self):
        all_params = []
        for m in self.get_modules_with_parameters():
            all_params.append(m.W)
            pass
        return all_params

    def load_state_dict(self, state_dict):
        assert len(state_dict) == len(self.get_modules_with_parameters())
        for m, lw in zip(self.get_modules_with_parameters(), state_dict):
            m.W = lw
            pass
        pass
    pass
