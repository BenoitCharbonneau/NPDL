import numpy as np
from utils.activations import get_activation


class LayerNorm:
    def __init__(self, num_class, eps=1e-5, momentum=0.9, weight_scale=None, activation='identity'):
        """
        Keyword Arguments:
            num_class {int} -- nombre de classes, soit le D dans (N, L, D)
            eps {float} -- hyperparamètre dans le calcul de normalisation (default: {1e-5})
            momentum {float} -- hyperparamètre qui détermine l'accumulation de moyenne et
                                de variance lors de l'entrainement dans le but d'être utilisées
                                lors des tests. (default: {0.9})
            weight_scale {float} -- écart type de la distribution normale utilisée
                                    pour l'initialisation des gammas. Si None,
                                    initialisation avec des 1. (default: {None})
            activation {str} -- identifiant de la fonction d'activation de la couche
                                (default: {'identite'})
        """

        self.num_class = num_class
        self.eps = eps
        self.momentum = momentum
        self.weight_scale = weight_scale
        self.activation_id = activation

        if weight_scale is not None:
            # Initialisation avec une distribution normale avec écart type = weight_scale
            self.gamma = np.random.normal(loc=0.0, scale=weight_scale, size=num_class)
        else:
            self.gamma = np.ones(num_class)

        self.beta = np.zeros(num_class)

        self.test_mean = np.zeros(num_class)
        self.test_var = np.ones(num_class)

        self.dgamma = 0
        self.dbeta = 0
        self.reg = 0.0
        self.cache = None

        self.activation = get_activation(activation)

    def forward_npdl(self, X, **kwargs):
        """Effectue la propagation avant.

        Arguments:
            X {ndarray} -- Input de la couche. Shape (N, L, D) ou (N, D)

        Keyword Arguments:
            **kwargs -- Utilisé pour indiquer si le forward
                        s'applique à l'entrainement ou au test (default: {'train'})

        Returns:
            ndarray -- Scores de la couche normalisés
        """
        mode = kwargs.get('mode', 'train')

        if len(X.shape) == 3:
            assert (self.num_class == X.shape[2])
            X_reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        elif len(X.shape) == 2:
            assert (self.num_class == X.shape[1])
            X_reshape = X
        else:
            raise Exception("Unsupported input shape")

        if mode == "train":
            batch_mean = np.mean(X_reshape, axis=1).reshape(-1, 1)
            batch_var = np.var(X_reshape, axis=1).reshape(-1, 1)
            delta = X_reshape - batch_mean
            sqrt_var = np.sqrt(batch_var + self.eps)
            batch_norm = delta / sqrt_var
            Z = self.gamma * batch_norm + self.beta
            self.cache = (delta, sqrt_var, Z)

            self.test_mean = self.momentum * self.test_mean + (1 - self.momentum) * np.mean(batch_mean, axis=0)
            self.test_var = self.momentum * self.test_var + (1 - self.momentum) * np.mean(batch_var, axis=0)

        elif mode == "test":
            batch_norm = (X_reshape - self.test_mean) / np.sqrt(self.test_var + self.eps)
            Z = self.gamma * batch_norm + self.beta
        else:
            raise Exception("Invalid forward mode %s" % mode)

        A = self.activation['forward'](Z)

        return A.reshape(X.shape)

    def backward_npdl(self, dA, **kwargs):
        """Effectue la rétro-propagation pour les paramètres de la
           couche.

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport au output de la couche.
                            Shape (N, L, D) ou (N, D)

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """

        if len(dA.shape) == 3:
            dA_reshape = dA.reshape(dA.shape[0] * dA.shape[1], self.num_class)
        elif len(dA.shape) == 2:
            dA_reshape = dA
        else:
            raise Exception("Unsupported input shape")

        N = dA_reshape.shape[1]
        delta, sqrt_var, Z = self.cache
        batch_norm = delta / sqrt_var

        self.dgamma = self.reg * self.gamma
        self.dbeta = self.reg * self.beta

        dZ = self.activation['backward'](Z) * dA_reshape

        dXn = dZ * self.gamma
        # dZ/dx = dXn * dXn/dx
        # En utilisant la règle des produits Xn = f*g =>
        # dZ/dx = dXn(df/dx*g + f*dg/dx) = (dXn * df/dx) * g + (dXn * f) * dg/dx
        f = delta
        dXnf = np.sum(dXn * f, axis=1)[:, np.newaxis]
        dXndf = -np.sum(dXn, axis=1)[:, np.newaxis] / N + dXn
        g = 1 / sqrt_var
        dvar = -2 * np.sum(f, axis=1)[:, np.newaxis] / N ** 2 + 2 * f / N
        dg = -1 / (2 * sqrt_var ** 3) * dvar
        dX = dXndf * g + dXnf * dg

        self.dgamma += np.sum(dA_reshape * batch_norm, axis=0)
        self.dbeta += np.sum(dA_reshape, axis=0)

        # Retourne la derivee du input de la couche (dX)
        return dX.reshape(dA.shape)

    def get_params(self, save=False):
        if save:
            return {'y': self.gamma, 'b': self.beta, 'm': self.test_mean, 'v': self.test_var}
        else:
            return {'y': self.gamma, 'b': self.beta}

    def set_param(self, param, value):
        if param == 'y':
            assert self.gamma.shape == value.shape
            self.gamma = value
        elif param == 'b':
            assert self.beta.shape == value.shape
            self.beta = value
        elif param == 'm':
            assert self.test_mean.shape == value.shape
            self.test_mean = value
        elif param == 'v':
            assert self.test_var.shape == value.shape
            self.test_var = value
        elif param == 'reg':
            self.reg = value
        else:
            raise Exception(param + " is not a valid parameter for LayerNorm layers")

    def get_gradients(self):
        return {'y': self.dgamma, 'b': self.dbeta}

    def reset(self):
        self.__init__(self.num_class,
                      self.eps, 
                      self.momentum, 
                      self.weight_scale, 
                      self.activation_id)
