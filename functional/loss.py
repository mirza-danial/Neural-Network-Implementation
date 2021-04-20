import numpy as np
from functional.constants import Constants

class LossFunctionFactory:
    @classmethod
    def get_loss_function(cls, loss):
        if loss == Constants.BinaryCrossEntropy:
            return cls.binary_cross_entropy

    @classmethod
    def binary_cross_entropy(cls, true_labels, predictions):
        m = true_labels.shape[1]

        logprobs = np.multiply(np.log(predictions), true_labels) + \
                   np.multiply((1 - true_labels), np.log(1 - predictions))

        cost = -1 / m * np.sum(logprobs)

        cost = np.squeeze(cost)

        return cost
