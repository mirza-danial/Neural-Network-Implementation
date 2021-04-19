import numpy as np


def initialize_weights(n_x, n_y, scale_factor=0.01):
    """
    Function to initialize random weights for a neural network
    :param n_x: size of input features to the layer
    :param n_y: size of output features to the layer
    :param scale_factor: scaling the weights to a specific value
    :return:weight matrix
    """
    weight_matrix = np.random.randn(n_y, n_x) * scale_factor
    return weight_matrix
