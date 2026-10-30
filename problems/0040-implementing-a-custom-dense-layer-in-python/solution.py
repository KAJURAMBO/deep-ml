
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        limit = 1 / np.sqrt(self.input_shape[0])

        self.W = np.random.uniform(
            -limit,
            limit,
            (self.input_shape[0], self.n_units)
        )

        self.w0 = np.zeros((1, self.n_units))

        self.W_optimizer = copy.deepcopy(optimizer)
        self.w0_optimizer = copy.deepcopy(optimizer)

    def parameters(self):
        return self.W.size + self.w0.size

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X @ self.W + self.w0

    def backward_pass(self, accum_grad):
        # Calculate input gradient using pre-update weights
        input_grad = accum_grad @ self.W.T

        if self.trainable:
            grad_W = self.layer_input.T @ accum_grad
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            self.W = self.W_optimizer.update(self.W, grad_W)
            self.w0 = self.w0_optimizer.update(self.w0, grad_w0)

        return input_grad

    def output_shape(self):
        return (self.n_units,)


# # Test
# dense_layer = Dense(n_units=3, input_shape=(2,))

# class MockOptimizer:
#     def update(self, weights, grad):
#         return weights - 0.01 * grad

# optimizer = MockOptimizer()
# dense_layer.initialize(optimizer)

# X = np.array([[1, 2]])
# output = dense_layer.forward_pass(X)

# accum_grad = np.array([[0.1, 0.2, 0.3]])
# back_output = dense_layer.backward_pass(accum_grad)

# print(np.round(back_output, 5).tolist())
