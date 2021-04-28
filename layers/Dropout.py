import numpy as np


class Dropout:
    def __init__(self, drop_rate=0.2):
        """
        Keyword Arguments:
            drop_rate {float} -- pourcentage de neurones qui ne sont pas activés
                                 à l'entrainement (default: {0.2})
        """
        self.drop_rate = drop_rate

        self.cache = None

    def forward_npdl(self, X, **kwargs):
        """Application du dropout inversé lors de la propagation avant.

        Arguments:
            X {ndarray} -- Outputs de la couche précédente.

        Keyword Arguments:
            **kwargs -- Utilisé pour indiquer si le forward
                        s'applique à l'entrainement ou au test
                        et pour inclure un seed (default: {'train', None})
        Returns:
            ndarray -- Scores de la couche
        """

        mode = kwargs.get('mode', 'train')
        seed = kwargs.get('seed', None)

        drop_mask = None
        
        if mode == 'train':
            if seed is not None:
                np.random.seed(seed)

            # On garde une valeur avec probabilité keep_rate (1 - drop_rate).
            drop_mask = np.random.choice([0, 1], size=X.shape, 
                                         p=[self.drop_rate, 1 - self.drop_rate])
            A = drop_mask * X / (1 - self.drop_rate)
        elif mode == 'test':
            A = X
        else:
            raise Exception("Invalid forward mode %s" % mode)

        self.cache = drop_mask

        return A

    def backward_npdl(self, dA, **kwargs):
        """Rétro-propagation pour la couche de dropout inversé.

        Arguments:
            dA {ndarray} -- Gradients de la loss par rapport aux sorties.

        Keyword Arguments:
            **kwargs -- Utilisé pour indiquer si le forward
                        s'applique à l'entrainement ou au test (default: {'train'})
        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """

        mode = kwargs.get('mode', 'train')

        if mode == 'train':
            drop_mask = self.cache

            dX = dA * drop_mask / (1 - self.drop_rate)
        elif mode == 'test':
            dX = dA
        else:
            raise Exception("Invalid forward mode %s" % mode)

        return dX

    def get_params(self, save=False):
        return {}

    def set_param(self, param, value):
        if param == 'drop_rate':
            self.drop_rate = drop_rate
        
    def get_gradients(self):
        return {}

    def reset(self):
        self.__init__(drop_rate=self.drop_rate)
