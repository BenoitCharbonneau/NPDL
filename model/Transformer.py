import numpy as np
from .Encoder import Encoder
from .Decoder import Decoder
from layers.Conv import Conv2DCython
from layers.Dense import Dense
from layers.Flatten import Flatten
from layers.TimeDistributed import TimeDistributed
from utils.positional_encoding import positional_encoding
from collections import OrderedDict


class TransformerEncoder:
    def __init__(self, num_stacks, length, d_model, d_ff, num_classes, num_heads=2):
        self.num_stacks = num_stacks
        self.length = length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.h = num_heads

        self.loss_function = None

        self.encoders = OrderedDict()
        self.conv = Conv2DCython(num_classes, filter_size=(length, d_model), weight_scale=1e-4)
        self.flatten = Flatten()
        self.cache = None
        for i in range(num_stacks):
            self.encoders["encoder" + str(i)] = Encoder(d_model, d_ff, num_heads=num_heads, id=i)

    def add_loss(self, loss_function):
        self.loss_function = loss_function

    def forward_npdl(self, X, **kwargs):
        """Propagation avant dans le modèle Transformer (many-to-one).

        Arguments:
            X {ndarray} -- Entrée du Transformer. Shape (N, L, d_model)

        Returns:
            ndarray -- Sortie du modèle. Shape (N, num_classes)
        """
        # Création du masque
        self.cache = np.ones(X.shape[:-1] + (1,))
        zero_indices = np.where((X == 0).all(axis=2))
        self.cache[zero_indices] = 0
        kwargs["output_mask"] = self.cache

        previous_output = positional_encoding(X, self.cache)
        for encoder in self.encoders.values():
            previous_output = encoder.forward_npdl(previous_output, **kwargs)

        previous_output = self.conv.forward_npdl(previous_output[:, np.newaxis, :, :], **kwargs)
        previous_output = self.flatten.forward_npdl(previous_output, **kwargs)

        return previous_output

    def backward_npdl(self, dOutput, **kwargs):
        """Rétropropagation dans le modèle Transformer (many-to-one).

        Arguments:
            dOutput {ndarray} -- Gradient de la perte par rapport à la 
                                 à la sortie du modèle. Shape (N, L, d_model)

        Returns:
            ndarray -- Gradient de la perte par rapport à l'entrée du modèle.
                       Shape (N, L, d_model)
        """

        kwargs["output_mask"] = self.cache
        dA = self.flatten.backward_npdl(dOutput, **kwargs)
        dA = self.conv.backward_npdl(dA, **kwargs)[:, 0, :, :]
        for encoder in reversed(list(self.encoders.values())):
            dA = np.sum(encoder.backward_npdl(dA, **kwargs), axis=0) / 3

        return dA

    def calculate_loss(self, model_output, targets, reg):
        for encoder in self.encoders.values():
            encoder.set_param('reg', reg)

        self.conv.set_param('reg', reg)

        return self.loss_function(model_output, targets, reg, self.parameters())

    def parameters(self):
        params = {}
        for name, encoder in self.encoders.items():
            params[name] = encoder.get_params()

        params['conv'] = self.conv.get_params()

        return params

    def gradients(self):
        gradients = {}
        for name, encoder in self.encoders.items():
            gradients[name] = encoder.get_gradients()

        gradients['conv'] = self.conv.get_gradients()

        return gradients

    def predict(self, X):
        scores = self.forward_npdl(X, mode='test')
        return np.argmax(scores, axis=len(scores.shape) - 1)

    def save_weights(self, filename):
        params = {}
        for name, layer in self.encoders.items():
            for key, item in layer.get_params(True).items():
                params[name + '_' + key] = item
        for key, item in self.conv.get_params(True).items():
            params["conv_" + key] = item
        np.savez_compressed(filename, **params)

    def load_weights(self, filename):
        model = np.load(filename + ".npz")
        for name in model.files:
            layer, param = name.split('_')
            if layer in self.encoders:
                self.encoders[layer].set_param(param, model[name])
            elif layer == "conv":
                self.conv.set_param(param, model[name])
            else:
                raise Exception("Layer: " + layer + " does not exist")

    def reset(self):
        self.__init__(self.num_stacks,
                      self.length,
                      self.d_model,
                      self.d_ff,
                      self.num_classes,
                      num_heads=self.h)


class Transformer:
    def __init__(self, num_stacks, length, d_model, d_ff, num_classes, num_heads=2):
        self.num_stacks = num_stacks
        self.length = length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.h = num_heads

        self.loss_function = None

        self.encoders = OrderedDict()
        self.decoders = OrderedDict()
        self.dense = TimeDistributed(Dense(d_model, num_classes, weight_scale=None), out_size=num_classes)
        self.cache = None

        for i in range(num_stacks):
            self.encoders["encoder" + str(i)] = Encoder(d_model, d_ff, num_heads=num_heads, id=i)
            self.decoders["decoder" + str(i)] = Decoder(length, d_model, d_ff, num_heads=num_heads, id=i)

    def add_loss(self, loss_function):
        self.loss_function = loss_function

    def forward_npdl(self, X, Y, **kwargs):
        """Propagation avant dans le modèle Transformer (many-to-many).

        Arguments:
            X {ndarray} -- Entrée du Transformer. Shape (N, L, d_model)
            Y {ndarray} -- Sortie attendue du Transformer. Shape (N, L, d_model)

        Returns:
            ndarray -- Sortie du modèle. Shape (N, L, d_model)
        """

        # Création du masque
        X_mask = np.ones(X.shape[:-1] + (1,))
        zero_indices = np.where((X == 0).all(axis=2))
        X_mask[zero_indices] = 0
        Y_mask = np.ones(Y.shape[:-1] + (1,))
        zero_indices = np.where((Y == 0).all(axis=2))
        Y_mask[zero_indices] = 0

        previous_outputX = positional_encoding(X, X_mask)
        previous_outputY = positional_encoding(Y, Y_mask)
        encoders = [encoder for encoder in self.encoders.values()]
        decoders = [decoder for decoder in self.decoders.values()]

        for i in range(self.num_stacks):
            kwargs["output_mask"] = X_mask
            previous_outputX = encoders[i].forward_npdl(previous_outputX, **kwargs)
            kwargs["encoder"] = previous_outputX
            kwargs["output_mask"] = Y_mask
            kwargs["attention_mask_E"] = X_mask
            previous_outputY = decoders[i].forward_npdl(previous_outputY, **kwargs)

        self.cache = (X_mask, Y_mask)

        return self.dense.forward_npdl(previous_outputY)

    def backward_npdl(self, dOutput, **kwargs):
        """Rétropropagation dans le modèle Transformer (many-to-many).

        Arguments:
            dOutput {ndarray} -- Gradient de la perte par rapport à la
                                 à la sortie du modèle. Shape (N, L, d_model)

        Returns:
            ndarray -- Gradient de la perte par rapport à l'entrée du modèle.
                       Shape (N, L, d_model)
            ndarray -- Gradient de la perte par rapport à la sortie attendue du modèle.
                       Shape (N, L, d_model)
        """

        X_mask, Y_mask = self.cache

        dA = self.dense.backward_npdl(dOutput)
        dE = None

        encoders = [encoder for encoder in self.encoders.values()]
        decoders = [decoder for decoder in self.decoders.values()]

        for i in range(self.num_stacks):
            kwargs["output_mask"] = Y_mask
            kwargs["attention_mask_E"] = X_mask

            dA, dE_partial = decoders[i].backward_npdl(dA, **kwargs)
            dA = np.sum(dA, axis=0) / 3
            if dE is None:
                dE = np.sum(dE_partial, axis=0) / 2
            else:
                dE += np.sum(dE_partial, axis=0) / 5

            kwargs["output_mask"] = X_mask
            dE = np.sum(encoders[i].backward_npdl(dE, **kwargs), axis=0) / 5

        return dA, dE

    def calculate_loss(self, model_output, targets, reg):
        for encoder in self.encoders.values():
            encoder.set_param('reg', reg)

        for decoder in self.decoders.values():
            decoder.set_param('reg', reg)

        self.dense.set_param('reg', reg)

        return self.loss_function(model_output, targets, reg, self.parameters())

    def parameters(self):
        params = {}
        for name, encoder in self.encoders.items():
            params[name] = encoder.get_params()
        for name, decoder in self.decoders.items():
            params[name] = decoder.get_params()

        params['dense'] = self.dense.get_params()

        return params

    def gradients(self):
        gradients = {}
        for name, encoder in self.encoders.items():
            gradients[name] = encoder.get_gradients()
        for name, decoder in self.decoders.items():
            gradients[name] = decoder.get_gradients()

        gradients['dense'] = self.dense.get_gradients()

        return gradients

    def predict(self, X, embedder, bos_id):
        """Méthode de prédiction pour le Transformer seq2seq. Contrairement
           à l'entraînement, le décodeur s'exécute de façon séquentielle
           en utilisant le token prédit à la position temporelle i - 1 comme
           entrée à la position i.

        Arguments:
            X {ndarray} -- Séquence en entrée. Shape (N, L, d_model)
            embedder {BPEmb} -- Embedder. Utilisé pour transformer les id en
                                vecteurs d'embeddings.
            bos_id {int} -- Index du bos dans le vocabulaire de l'embedder.

        Returns:
            ndarray -- Ids des prédictions. Shape (N, L)
        """

        N = len(X)
        L = X.shape[1]
        
        predictions = np.zeros((N, L))

        Y = np.zeros(X.shape)
        Y[:, 0] = embedder.emb.vectors[bos_id]
        
        for pos in range(L):
            prediction_ids = np.argmax(self.forward_npdl(X, Y, mode='test')[:, pos], axis=1).reshape(N, 1)
            Y[:, pos] = np.apply_along_axis(self.embed, 1, prediction_ids, embedder).reshape(N, -1)
            predictions[:, pos] = prediction_ids.reshape(N,)
            
        return predictions.astype(int)

    def embed(self, ids, embedder):
        return embedder.emb.vectors[ids.tolist()]

    def save_weights(self, filename):
        params = {}
        for name, layer in self.encoders.items():
            for key, item in layer.get_params(True).items():
                params[name + '_' + key] = item
        for name, layer in self.decoders.items():
            for key, item in layer.get_params(True).items():
                params[name + '_' + key] = item
        for key, item in self.dense.get_params(True).items():
            params["dense_" + key] = item
        np.savez_compressed(filename, **params)

    def load_weights(self, filename):
        model = np.load(filename + ".npz")
        for name in model.files:
            layer, param = name.split('_')
            if layer in self.encoders:
                self.encoders[layer].set_param(param, model[name])
            elif layer in self.decoders:
                self.decoders[layer].set_param(param, model[name])
            elif layer == "dense":
                self.dense.set_param(param, model[name])
            else:
                raise Exception("Layer: " + layer + " does not exist")

    def reset(self):
        self.__init__(self.num_stacks,
                      self.length,
                      self.d_model,
                      self.d_ff,
                      self.num_classes,
                      num_heads=self.h)
