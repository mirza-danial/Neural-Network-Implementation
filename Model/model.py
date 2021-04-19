import numpy as np
from Layers.layers import Layer
from tqdm import  tqdm


class SequentialModel:

    def __init__(self, layers=None):

        if layers is None:
            self.layers = list()
        else:
            self.layers = layers

        assert all([isinstance(layer, Layer) for layer in self.layers]), "Unknown layer found"

    def add(self, layer):
        if isinstance(layer, Layer):
            self.layers.append(layer)
        else:
            raise Exception("Incompatible type passed to the function")

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, loss_gradient):
        gradients = list()
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)
            if isinstance(loss_gradient, tuple):
                gradients.append(loss_gradient[1:])
                loss_gradient = loss_gradient[0]

        return reversed(gradients)

    def optimize(self, gradients, learning_rate):
        optimizable_layers = [layer for layer in self.layers if layer.is_trainable]
        for layer, gradient in zip(optimizable_layers, gradients):
            if layer.is_trainable:
                layer.weights = layer.weights - learning_rate * gradient[0]
                layer.bias = layer.bias - learning_rate * gradient[1]

    def fit(self, x_train, y_train, epochs=10, learning_rate = 0.0075, verbos = 1):
        pbar = tqdm(range(epochs))
        for i in pbar:
            y_predict = self.forward(x_train)

            m = y_train.shape[1]
            logprobs = np.multiply(np.log(y_predict), y_train) + np.multiply((1 - y_train), np.log(1 - y_predict))
            cost = -1 / m * np.sum(logprobs)
            cost = np.squeeze(cost)

            gradient = - (np.divide(y_train, y_predict) - np.divide(1 - y_train, 1 - y_predict))

            gradients = self.backward(gradient)

            self.optimize(gradients, learning_rate)

            if verbos == 1 and i % 100 == 0:
                pbar.set_description("Cost after iteration {}: {}".format(i, np.squeeze(cost)))





