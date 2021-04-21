from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import abc
from Model.constants import Constants


class MetricFactory:

    @classmethod
    def get_metric(cls, metric_name):
        if metric_name == Constants.ACCURACY_SCORE:
            return Accuracy()
        elif metric_name == Constants.PRECISION_SCORE:
            return Precision()
        elif metric_name == Constants.RECALL_SCORE:
            return Recall()
        elif metric_name == Constants.F1_SCORE:
            return F1Score()
        else:
            raise Exception('Invalid or non-existent evaluation metric found')


class AbstractMetric:
    @abc.abstractmethod
    def compute(self, y_predict, y_actual):
        pass


class Accuracy(AbstractMetric):

    def compute(self, y_predict, y_actual):
        return accuracy_score(y_actual.squeeze(), y_predict.squeeze())


class Precision(AbstractMetric):

    def compute(self, y_predict, y_actual):
        return precision_score(y_actual.squeeze(), y_predict.squeeze())


class Recall(AbstractMetric):

    def compute(self, y_predict, y_actual):
        return recall_score(y_actual.squeeze(), y_predict.squeeze())


class F1Score(AbstractMetric):

    def compute(self, y_predict, y_actual):
        return accuracy_score(y_actual.squeeze(), y_predict.squeeze())
