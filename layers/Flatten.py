import numpy as np


class Flatten:
    def forward_npdl(self, X, **kwargs):
        self.cache = X.shape
        N = X.shape[0]

        return X.reshape(N, -1)

    def backward_npdl(self, dA, **kwargs):
        return dA.reshape(self.cache)

    def set_param(self, param, value):
        pass

    def get_params(self, save=False):
        return {}

    def get_gradients(self):
        return {}

    def reset(self):
        pass
