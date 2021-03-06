import numpy as np
from Layers.layers import Layer
from tqdm import tqdm
from functional.loss import LossFunctionFactory
from Model.metrics import MetricFactory

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

    def fit(self, x_train, y_train, loss, epochs=10, learning_rate=0.0075, verbose=1,
            x_eval=None, y_eval=None, metrics=None):
        pbar = tqdm(range(epochs))
        history = {}
        for i in pbar:
            epoch_history = dict()

            y_predict = self.forward(x_train)

            loss_function = LossFunctionFactory.get_loss_function(loss)
            cost = loss_function(true_labels=y_train, predictions=y_predict)

            gradient = - (np.divide(y_train, y_predict) - np.divide(1 - y_train, 1 - y_predict))

            gradients = self.backward(gradient)

            self.optimize(gradients, learning_rate)

            train_history = self.test(x_train, y_train, metrics)
            train_history = {"train_" + key: value for key,value in train_history.items()}
            train_history['train_loss'] = cost
            epoch_history.update(train_history)

            eval_history = dict()
            if x_eval is not None and y_eval is not None :
                eval_predict = self.forward(x_eval)
                eval_loss = loss_function(true_labels=y_eval, predictions=eval_predict)
                eval_history = self.test(x_eval, y_eval, metrics)
                eval_history = {"eval_" + key: value for key, value in eval_history.items()}
                eval_history['eval_loss'] = eval_loss
                epoch_history.update(eval_history)



            if verbose == 1 and i % 100 == 0:
                history[i] = epoch_history
                pbar.set_description("cost after iteration {}: {}".format(i, cost))

        return history

    def predict(self, x):
        return (self.forward(x) > 0.5).astype(int)

    def test(self, x, y, metrics):

        y_predict = self.predict(x)

        evaluations = dict()
        if isinstance(metrics, list):
            for metric_name in metrics:
                metric = MetricFactory.get_metric(metric_name)
                result = metric.compute(y_predict, y)
                evaluations[metric_name] = result

        elif isinstance(metrics, str):
            metric = MetricFactory.get_metric(metrics)
            result = metric.compute(y_predict, y)
            evaluations[metric] = result

        return evaluations

