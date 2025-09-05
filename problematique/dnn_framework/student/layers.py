import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        super().__init__()
        self.input_count = input_count
        self.output_count = output_count
        self.parameters = {
            "w": np.random.randn(output_count, input_count), "b": np.zeros((1,output_count))
        }

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return {}

    def forward(self, x):
        W = self.parameters["w"]
        b = self.parameters["b"]

        return x @ W.T + b, x

    def backward(self, output_grad, cache):
        W = self.parameters["w"]

        dx = output_grad @ W
        dw = output_grad.T @ cache
        db = output_grad.sum(axis=0, keepdims=True)

        return dx, {"w": dw, "b": db}


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.input_count = input_count
        self.eps = 1e-7
        
        # gamma: controls the scaling
        # beta: controls the shifting
        self.parameters = {
            "gamma": np.ones((1,input_count)), "beta": np.zeros((1,input_count))
        }
        
        self.buffers = {
            "global_mean": np.zeros(input_count), "global_variance": np.zeros(input_count)
        }

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return self.buffers

    def forward(self, x):
        if self._is_training:
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):
        gamma = self.parameters["gamma"]
        beta = self.parameters["beta"]
        global_mean = self.buffers["global_mean"]
        global_variance = self.buffers["global_variance"]
        
        batch_mean = np.mean(x, axis=0, keepdims=True)
        batch_var = np.var(x, axis=0, keepdims=True)
        
        x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps) # new x
        y = gamma * x_hat + beta # output based hyperparams and new x (scaling and shifting)
        
        # not sure of this
        self.buffers["global_mean"] = self.alpha * batch_mean + (1 - self.alpha) * global_mean
        self.buffers["global_variance"] = self.alpha * batch_var + (1 - self.alpha) * global_variance
        
        cache = (x, x_hat, batch_mean, batch_var)
        
        return y, cache

    def _forward_evaluation(self, x):
        gamma = self.parameters["gamma"]
        beta = self.parameters["beta"]
        global_mean = self.buffers["global_mean"]
        global_variance = self.buffers["global_variance"]
        
        x_hat = (x - global_mean) / np.sqrt(global_variance + self.eps)
        y = gamma * x_hat + beta
        
        return y, None

    def backward(self, output_grad, cache):
        x, x_hat, mean, var = cache
        N, D = x.shape
        
        # TODO: à comprendre 
        dgamma = np.sum(output_grad * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(output_grad, axis=0, keepdims=True)
        
        dx_hat = output_grad * self.parameters["gamma"]
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)

        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N

        grads = {"gamma": dgamma, "beta": dbeta}
        return dx, grads


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """
    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        return 1 / (1 + np.exp(-x)), x

    def backward(self, output_grad, cache):
        forward, cache = self.forward(cache)
        return output_grad * forward * (1 - forward), {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """
    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        return np.where(x < 0, 0, x), x

    def backward(self, output_grad, cache):
        return np.where(cache < 0, 0, 1) * output_grad, {}
