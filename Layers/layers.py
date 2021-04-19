import numpy as np
from Layers.utils import initialize_weights
import abc


class Layer:
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def backward(self, x):
        pass


class Linear(Layer):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.in_features = input_features
        self.out_features = output_features
        self.is_trainable = True

        self.weights = initialize_weights(self.in_features, self.out_features)
        self.bias = initialize_weights(1,self.out_features)
        self.cache = None

    def forward(self, input):
        output = None
        try:
            # Check if input dimension matches the required ones
            assert input.shape[0] == self.in_features, "Mismatch input dimensions"

            output = np.dot(self.weights, input) + self.bias

            # Cache the input or gradient calculation
            self.cache = input

        except Exception as exp:
            raise exp

        return output

    def backward(self, gradient):
        output = None
        try:
            dW = (1 / self.in_features) * np.dot(gradient, self.cache.T)
            db = (1 / self.in_features) * (np.sum(gradient, axis=1, keepdims=True))
            dA = np.dot(self.weights.T, gradient)

            output = (dA, dW, db)
        except Exception as exp:
            raise exp
        return output


class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.is_trainable = False
        self.cache = None

    def forward(self, input):
        output = None
        try:
            output = 1 / (1 + np.exp(-input))
            self.cache = input

        except Exception as exp:
            raise exp

        return output

    def backward(self, gradient):
        output = None
        try:
            s = 1 / (1 + np.exp(-self.cache))
            dZ = gradient * s * (1 - s)

            assert (dZ.shape == self.cache.shape)

            output = dZ

        except Exception as exp:
            raise exp

        return output


class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.is_trainable = False
        self.cache = None

    def forward(self, input):
        output = None
        try:
            output = np.maximum(0, input)
            self.cache = input

        except Exception as exp:
            raise exp

        return output

    def backward(self, gradient):
        output = None
        try:
            dZ = np.array(gradient, copy=True)  # just converting dz to a correct object.

            # When input <= 0, you should set dz to 0 as well.
            dZ[self.cache <= 0] = 0

            output = dZ

        except Exception as exp:
            raise exp

        return output
