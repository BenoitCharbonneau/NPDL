
class TimeDistributed:
    
    def __init__(self, layer, out_size=1, n_dims=1):
        """Initialisation de la couche TimeDistributed

        Arguments:
            layer {Dense} -- Couche dense en entrée

        Keyword Arguments:
            out_size {int, tuple} -- dimension d'un vecteur d'une
                                     séquence en sortie. Si tuple,
                                     doit contenir n_dims éléments (default: {1})
            n_dims {int} -- nombre de dimensions d'un vecteur d'une
                            séquence en entrée (default: {1})
        """

        self.layer = layer

        if isinstance(out_size, int):
            self.out_size = (out_size,)
        else:
            self.out_size = out_size

        self.n_dims = n_dims

        if len(self.out_size) is not self.n_dims:
            raise Exception("Invalid out_size shape")

        self.in_dims = None

    def forward_npdl(self, X, **kwargs):
        """Propagation avant dans une couche distribuée.

        Arguments:
            X {ndarray} -- Entrée n dimensions de la couche (3 ou plus)
        """

        s = (-1,) + X.shape[-self.n_dims:]

        inputs = X.reshape(s)

        outputs = self.layer.forward_npdl(inputs)
    
        self.in_dims = X.shape[-self.n_dims:]

        return outputs.reshape(X.shape[:-self.n_dims] + self.out_size)

    def backward_npdl(self, dA, **kwargs):
        """Rétro-propagation dans la couche distribuée.

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport à la sortie
                            de la couche, n dimensions (3 ou plus)
        """
        
        s = (-1,) + dA.shape[-self.n_dims:]

        dOutputs = dA.reshape(s)

        dInputs = self.layer.backward_npdl(dOutputs)

        return dInputs.reshape(dA.shape[:-self.n_dims] + self.in_dims)

    def get_params(self, save=False):
        return self.layer.get_params()

    def set_param(self, param, value):
        self.layer.set_param(param, value)

    def get_gradients(self):
        return self.layer.get_gradients()

    def reset(self):
        self.layer.reset()
