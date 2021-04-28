import math
import numpy as np
from .utils.LSTMCell import LSTMCell


class LSTM:

    def __init__(self, length, input_size, hidden_size, single_output=True, weight_scale=1e-4, mask_zeros=True):
        """
        Arguments:
            length {int} -- nombre de cellules dans la couche, soit le L dans (N, L, H).
            input_size {int} -- la taille de l'entrée dans la couche, soit le H dans (N, L, H).
            hidden_size {int} -- la taille de la sortie dans la couche (et de la cellule mémoire).
            single_output {bool} -- détermine si la sortie est unique ou de taille L.

        Keyword Arguments:
            weight_scale {float} -- Facteur de variance des poids (default: {1e-4})
        """

        self.cell_list = []
        self.length = length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.single_output = single_output
        self.weight_scale = weight_scale
        self.mask_zeros = mask_zeros

        self.Wc_i = self.init_weights(input_size, hidden_size, weight_scale)
        self.Wu_i = self.init_weights(input_size, hidden_size, weight_scale)
        self.Wf_i = self.init_weights(input_size, hidden_size, weight_scale)
        self.Wo_i = self.init_weights(input_size, hidden_size, weight_scale)

        self.Wc_h = self.init_weights(hidden_size, hidden_size, weight_scale)
        self.Wu_h = self.init_weights(hidden_size, hidden_size, weight_scale)
        self.Wf_h = self.init_weights(hidden_size, hidden_size, weight_scale)
        self.Wo_h = self.init_weights(hidden_size, hidden_size, weight_scale)

        self.bc = np.zeros(hidden_size)
        self.bu = np.zeros(hidden_size)
        self.bf = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)

        self.init_gradients()

        self.weights = {'Wc_i': self.Wc_i,
                        'Wu_i': self.Wu_i,
                        'Wf_i': self.Wf_i,
                        'Wo_i': self.Wo_i,
                        'Wc_h': self.Wc_h,
                        'Wu_h': self.Wu_h,
                        'Wf_h': self.Wf_h,
                        'Wo_h': self.Wo_h}

        self.biases = {'bc': self.bc,
                       'bu': self.bu,
                       'bf': self.bf,
                       'bo': self.bo}

        self.reg = 0.0
        
        for i in range(length):
            self.cell_list.append(LSTMCell(input_size, hidden_size, self.weights, self.biases))

    def init_weights(self, input_size, hidden_size, weight_scale=1e-4):
        """Méthode d'initialisation d'une matrice de poids de la cellule LSTM.

        Arguments:
            input_size {int} -- Taille d'une donnée en entrée
            hidden_size {int} -- Taille cachée de la cellule

        Keyword Arguments:
            weight_scale {float} -- facteur de normalisation des poids (default: {1e-4})
            init_type {str} -- type d'initialisation (None, xavier, he) (default: {None})

        Raises:
            Exception: Type d'initialisation invalide

        Returns:
            ndarray -- matrice de poids de dimensions (input_size + hidden_size, hidden_size)
        """
        if weight_scale is not None:
            return np.random.normal(loc=0.0, scale=weight_scale, size=(input_size, hidden_size))
        else:
            return np.random.normal(loc=0.0, scale=math.sqrt(2.0 / (input_size + hidden_size)),
                                    size=(input_size, hidden_size))

    def init_gradients(self):
        self.dWc_i = np.zeros(self.Wc_i.shape)
        self.dWu_i = np.zeros(self.Wu_i.shape)
        self.dWf_i = np.zeros(self.Wf_i.shape)
        self.dWo_i = np.zeros(self.Wo_i.shape)

        self.dWc_h = np.zeros(self.Wc_h.shape)
        self.dWu_h = np.zeros(self.Wu_h.shape)
        self.dWf_h = np.zeros(self.Wf_h.shape)
        self.dWo_h = np.zeros(self.Wo_h.shape)

        self.dbc = np.zeros(self.bc.shape)
        self.dbu = np.zeros(self.bu.shape)
        self.dbf = np.zeros(self.bf.shape)
        self.dbo = np.zeros(self.bo.shape)

    def forward_npdl(self, X, **kwargs):
        """Effectue la propagation avant.

        Arguments:
            X {ndarray} -- Input de la couche. Shape (N, length, input_size)

        Returns:
            ndarray -- Scores de la couche. Shape (N, if [single_output] 1 else length, hidden_size)
        """
        if len(X.shape) == 3:
            H = X
        elif len(X.shape) == 2:
            H = X[np.newaxis, :]
        else:
            raise Exception("Unsupported shape: " + X.shape)

        assert H.shape[1] == self.length

        H_prev = np.zeros((H.shape[0], self.hidden_size))
        c_prev = np.zeros((H.shape[0], self.hidden_size))
        outputs = np.empty((H.shape[0], self.length, self.hidden_size))
        for i, cell in enumerate(self.cell_list):
            c_prev, H_prev, H_up = cell.forward_npdl(H[:, i, :], H_prev, c_prev, mask_zeros=self.mask_zeros, **kwargs)
            if not self.single_output:
                outputs[:, i, :] = H_up

        if self.single_output:
            return H_prev[:, np.newaxis, :]
        else:
            return outputs

    def backward_npdl(self, dA, **kwargs):
        """Effectue la rétro-propagation pour les paramètres de la
           couche.

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport au output de la couche.
                            Shape (N, if [single_output] 1 else length, hidden_size)

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
                       Shape (N, length, input_size)
        """
        assert len(dA.shape) == 3
        if self.single_output:
            assert dA.shape[1] == 1
        else:
            assert dA.shape[1] == self.length

        # Réinitialiser les gradients
        self.init_gradients()
        
        dH_partial = np.zeros((dA.shape[0], self.hidden_size))
        dC_partial = np.zeros((dA.shape[0], self.hidden_size))
        dX = np.empty((dA.shape[0], self.length, self.input_size))

        for i, cell in enumerate(reversed(self.cell_list)):
            index = self.length - 1 - i
            if self.single_output:
                if index < self.length - 1:
                    dH_up = np.zeros((dA.shape[0], self.hidden_size))
                else:
                    dH_up = dA[:, 0, :]
            else:
                dH_up = dA[:, index, :]
            dX[:, index, :], dH_partial, dC_partial = cell.backward_npdl(dH_up, dH_partial, dC_partial, **kwargs)

            # Additionner les gradients des poids et des biais partiels de chaque cellule
            partial_gradients = cell.get_gradients()

            self.dWc_i += partial_gradients['Wc_i']
            self.dWu_i += partial_gradients['Wu_i']
            self.dWf_i += partial_gradients['Wf_i']
            self.dWo_i += partial_gradients['Wo_i']

            self.dWc_h += partial_gradients['Wc_h']
            self.dWu_h += partial_gradients['Wu_h']
            self.dWf_h += partial_gradients['Wf_h']
            self.dWo_h += partial_gradients['Wo_h']

            self.dbc += partial_gradients['bc']
            self.dbu += partial_gradients['bu']
            self.dbf += partial_gradients['bf']
            self.dbo += partial_gradients['bo']

        # Appliquer la régularisation sur les poids et les biais
        self.dWc_i += self.reg * self.Wc_i
        self.dWu_i += self.reg * self.Wu_i
        self.dWf_i += self.reg * self.Wf_i
        self.dWo_i += self.reg * self.Wo_i

        self.dWc_h += self.reg * self.Wc_h
        self.dWu_h += self.reg * self.Wu_h
        self.dWf_h += self.reg * self.Wf_h
        self.dWo_h += self.reg * self.Wo_h

        self.dbc += self.reg * self.bc
        self.dbu += self.reg * self.bu
        self.dbf += self.reg * self.bf
        self.dbo +=self.reg * self.bo

        return dX

    def get_params(self, save=False):
        return {'Wci': self.Wc_i,
                'Wui': self.Wu_i,
                'Wfi': self.Wf_i,
                'Woi': self.Wo_i,
                'Wch': self.Wc_h,
                'Wuh': self.Wu_h,
                'Wfh': self.Wf_h,
                'Woh': self.Wo_h,
                'bc': self.bc,
                'bu': self.bu,
                'bf': self.bf,
                'bo': self.bo}

    def set_param(self, param, value):
        if param == 'Wci':
            assert self.Wc_i.shape == value.shape
            self.Wc_i = value
        elif param == 'Wui':
            assert self.Wu_i.shape == value.shape
            self.Wu_i = value
        elif param == 'Wfi':
            assert self.Wf_i.shape == value.shape
            self.Wf_i = value
        elif param == 'Woi':
            assert self.Wo_i.shape == value.shape
            self.Wo_i = value
        elif param == 'Wch':
            assert self.Wc_h.shape == value.shape
            self.Wc_h = value
        elif param == 'Wuh':
            assert self.Wu_h.shape == value.shape
            self.Wu_h = value
        elif param == 'Wfh':
            assert self.Wf_h.shape == value.shape
            self.Wf_h = value
        elif param == 'Woh':
            assert self.Wo_h.shape == value.shape
            self.Wo_h = value
        elif param == 'bc':
            assert self.bc.shape == value.shape
            self.bc = value
        elif param == 'bu':
            assert self.bu.shape == value.shape
            self.bu = value
        elif param == 'bf':
            assert self.bf.shape == value.shape
            self.bf = value
        elif param == 'bo':
            assert self.bo.shape == value.shape
            self.bo = value
        elif param == 'reg':
            self.reg = value
        else:
            raise Exception(param + " is not a valid parameter for LSTM")

    def get_gradients(self):
        return {'Wci': self.dWc_i,
                'Wui': self.dWu_i,
                'Wfi': self.dWf_i,
                'Woi': self.dWo_i,
                'Wch': self.dWc_h,
                'Wuh': self.dWu_h,
                'Wfh': self.dWf_h,
                'Woh': self.dWo_h,
                'bc': self.dbc,
                'bu': self.dbu,
                'bf': self.dbf,
                'bo': self.dbo}

    def reset(self):
        self.__init__(self.length,
                      self.input_size,
                      self.hidden_size,
                      single_output=self.single_output,
                      weight_scale=self.weight_scale,
                      mask_zeros=self.mask_zeros)
