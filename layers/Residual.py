import copy


class Residual:
    def __init__(self, layers):
        self.layers = layers

    def forward_npdl(self, X, **kwargs):
        """Propagation avant au travers d'une couche résiduelle.

        Arguments:
            X {ndarray} -- Entrée de la couche.

        Returns:
            ndarray -- Sortie résiduelle de la couche.
        """

        Q = kwargs.get('Q', None)
        K = kwargs.get('K', None)

        clean_kwargs = copy.deepcopy(kwargs)
        _ = clean_kwargs.pop('Q', None)
        _ = clean_kwargs.pop('K', None)

        previous_output = X
        for layer_name, layer in self.layers.items():
            if layer_name == 'multihead1':
                assert Q is not None
                assert K is not None
                previous_output = layer.forward_npdl(Q, K, previous_output, **clean_kwargs)
            else:
                previous_output = layer.forward_npdl(previous_output, **kwargs)

        return previous_output + X

    def backward_npdl(self, dOutput, **kwargs):
        """Rétropropagation dans la couche résiduelle.

        Arguments:
            dA {ndarray} -- Dérivée de la perte en fonction 
                            de l'entrée de la couche.
        """

        dA = dOutput
        for layer in reversed(list(self.layers.values())):
            dA = layer.backward_npdl(dA, **kwargs)

        return dA + dOutput

    def get_params(self, save=False):
        params = {}
        for layer_name, layer in self.layers.items():
            layer_params = layer.get_params()
            layer_params = {layer_name + '-' + k: v for k, v in layer_params.items()}
            params = {**params, **layer_params}

        return params

    def set_param(self, param, value):
        if param == 'reg':
            for layer in self.layers.values():
                layer.set_param(param, value)
        else:
            layer_name = param.split('-')[0]
            name = '-'.join(param.split('-')[1:])
            self.layers[layer_name].set_param(name, value)

    def get_gradients(self):
        gradients = {}
        for layer_name, layer in self.layers.items():
            layer_gradients = layer.get_gradients()
            layer_gradients = {layer_name + '-' + k: v for k, v in layer_gradients.items()}
            gradients = {**gradients, **layer_gradients}

        return gradients

    def reset(self):
        self.__init__(self.layers)
