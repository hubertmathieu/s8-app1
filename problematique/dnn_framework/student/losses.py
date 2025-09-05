import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        probs = softmax(x)
        N = x.shape[0]

        batch_indices = np.arange(N) # Example: if N=3 → [0, 1, 2]
        target_class_probs = probs[batch_indices, target] # target ex: [0, 0, 1], only need the 1
        safe_probs = target_class_probs + 1e-15 # prevent log(0)
        log = -np.log(safe_probs) # not using target because it's not one hot encoded
        loss = np.sum(log)/N

        grad = probs.copy()
        grad[batch_indices, target] -= 1
        grad /= N

        return loss, grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        N = x.size  # total number of elements
        loss = np.sum((target - x) ** 2) / N
        grad = -2 * (target - x) / N  # gradient w.r.t x
        return loss, grad
