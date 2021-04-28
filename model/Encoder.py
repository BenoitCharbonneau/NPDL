import numpy as np
from layers.LayerNorm import LayerNorm
from layers.Residual import Residual
from layers.Dense import Dense
from layers.Attention import MultiHeadAttention
from layers.Dropout import Dropout
from layers.TimeDistributed import TimeDistributed
from collections import OrderedDict


class Encoder:
    def __init__(self, d_model, d_ff, num_heads=2, id=0):
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = num_heads
        self.id = id

        self.layers = OrderedDict()
        self.layers["residual1"] = Residual(
            {'multihead1': MultiHeadAttention(d_model, num_heads=num_heads, weight_scale=None),
             'dropout1': Dropout(0.1)})
        self.layers["layernorm1"] = LayerNorm(d_model)
        self.layers["residual2"] = Residual(
            {'dense1': TimeDistributed(Dense(d_model, d_ff, activation="relu", weight_scale=None), out_size=d_ff),
             'dense2': TimeDistributed(Dense(d_ff, d_model, weight_scale=None), out_size=d_model),
             'dropout2': Dropout(0.1)})
        self.layers["layernorm2"] = LayerNorm(d_model)

    def forward_npdl(self, X, **kwargs):
        """Propagation avant dans le bloc encodeur du Transformer.

        Arguments:
            X {ndarray} -- Données en entrée. Shape (N, L, d_model)

        Returns:
            ndarray -- Données en sortie du bloc. Shape (N, L, d_model)
        """

        previous_output = X

        output_mask = kwargs.get('output_mask', np.ones(X.shape[:-1] + (1,)))

        for layer_name, layer in self.layers.items():
            if layer_name == 'residual1':
                previous_output = layer.forward_npdl(previous_output, Q=previous_output, K=previous_output, **kwargs)
            elif layer_name == 'residual2':
                previous_output = layer.forward_npdl(previous_output, **kwargs) * output_mask
            else:
                previous_output = layer.forward_npdl(previous_output, **kwargs)

        return previous_output

    def backward_npdl(self, dOutput, **kwargs):
        """Rétropropagation dans le bloc encodeur du Transformer.

        Arguments:
            dOutput {ndarray} -- Gradient de la perte par rapport à la 
                                 sortie de l'encodeur. Shape (N, L, d_model)

        Returns:
            ndarray -- Gradient de la perte par rapport à l'entrée de l'encodeur.
                       Shape (N, L, d_model)
        """

        dA = dOutput

        output_mask = kwargs.get('output_mask', np.ones(dOutput.shape[:-1] + (1,)))

        for layer_name, layer in reversed(list(self.layers.items())):
            if layer_name == 'residual2':
                dA = dA * output_mask

            dA = layer.backward_npdl(dA, **kwargs)

        return dA

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
                layer.set_param('reg', value)
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
        self.__init__(self.d_model,
                      self.d_ff,
                      num_heads=self.h,
                      id=self.id)
