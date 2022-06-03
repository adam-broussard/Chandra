"""
Contains any custom metrics needed for model evaluation.
"""
# pylint: disable=[E0611,E0401]
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
# pylint: enable-[E0611,E0401]


class F1_Score(tf.keras.metrics.Metric):
    """
    A tensorflow metric for calculating the F1 score.

    Args:
        name (string): A custom name for this metric
    """

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = Precision(thresholds=0.5)
        self.recall_fn = Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred):
        """
        Update the object state

        Args:
            y_true (iterable): The true target values
            y_pred (iterable): The predicted target values
        """
        p = self.precision_fn(y_true, y_pred, sample_weight=sample_weight)
        r = self.recall_fn(y_true, y_pred, sample_weight=sample_weight)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        """
        Returns the result.
        """
        return self.f1

    def reset_states(self):
        """
        Resets all states to zero values.
        """
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)
