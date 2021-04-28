import numpy as np
from utils.activations import sigmoid_forward_npdl, sigmoid_backward_npdl, tanh_forward_npdl, tanh_backward_npdl


class LSTMCell:

    def __init__(self, input_size, hidden_size, weights, biases):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wc_i = weights['Wc_i']
        self.Wu_i = weights['Wu_i']
        self.Wf_i = weights['Wf_i']
        self.Wo_i = weights['Wo_i']

        self.Wc_h = weights['Wc_h']
        self.Wu_h = weights['Wu_h']
        self.Wf_h = weights['Wf_h']
        self.Wo_h = weights['Wo_h']

        self.bc = biases['bc']
        self.bu = biases['bu']
        self.bf = biases['bf']
        self.bo = biases['bo']

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

        self.cache = None

    def forward_npdl(self, X, H_prev, c_prev, mask_zeros=True, **kwargs):
        """Propagation avant dans la cellule LSTM

        Arguments:
            X {ndarray} -- Entrée de la cellule, 
                           shape (N, D)
            H_prev {ndarray} -- Entrée cachée de la cellule,
                                shape (N, H)
            c_prev {ndarray} -- cellule mémoire du réseau LSTM,
                                shape (N, H)
        """

        # Calcul de c_temp
        Zc_h = H_prev.dot(self.Wc_h)
        Zc_i = X.dot(self.Wc_i)
        self.c_temp = tanh_forward_npdl(Zc_h + Zc_i + self.bc)

        # Calcul de la update gate
        Zu_h = H_prev.dot(self.Wu_h)
        Zu_i = X.dot(self.Wu_i)
        self.U = sigmoid_forward_npdl(Zu_h + Zu_i + self.bu)

        # Calcul de la forget gate
        Zf_h = H_prev.dot(self.Wf_h)
        Zf_i = X.dot(self.Wf_i)
        self.F = sigmoid_forward_npdl(Zf_h + Zf_i + self.bf)

        # Calcul de la output gate
        Zo_h = H_prev.dot(self.Wo_h)
        Zo_i = X.dot(self.Wo_i)
        self.O = sigmoid_forward_npdl(Zo_h + Zo_i + self.bo)

        self.C = self.U * self.c_temp + self.F * c_prev
        self.H = self.O * tanh_forward_npdl(self.C)
        self.H_up = np.copy(self.H)

        m_temp = np.where((X == 0).all(axis=1))

        if mask_zeros and len(m_temp[0]) > 0:
            mask = m_temp

            self.C[mask] = c_prev[mask]
            self.H[mask] = H_prev[mask]
            self.H_up[mask] = np.zeros((self.hidden_size))
        else:
            mask = None

        self.cache = (X, H_prev, c_prev, mask)

        return self.C, self.H, self.H_up

    def backward_npdl(self, dH_up, dH_partial, dC_partial, **kwargs):
        """Rétro-propagation dans la cellule LSTM

        Arguments:
            dH_up {ndarray} -- Dérivée de la loss en fonction de h arrivant du haut,
                               shape (N, H).
            dH_partial {ndarray} -- Dérivée de la loss en fonction de h arrivant de
                                    la cellule suivante, shape (N, H).
            dC_partial {ndarray} -- Dérivée de la loss en fonction de c arrivant de
                                    la cellule suivante, shape (N, H).

        Returns:
            tuple -- tuple contenant les dérivées a passer à la cellule précédente.
        """

        X, H_prev, c_prev, mask = self.cache

        if mask is not None:
            dH_partial_masked = np.copy(dH_partial)
            dH_partial_masked[mask] = np.zeros(self.hidden_size)

            dH_up_masked = np.copy(dH_up)
            dH_up_masked[mask] = np.zeros(self.hidden_size)

            dC_partial_masked = np.copy(dC_partial)
            dC_partial_masked[mask] = np.zeros(self.hidden_size)

            dH = dH_up_masked + dH_partial_masked
            dC = dC_partial_masked + dH * self.O * tanh_backward_npdl(self.C)
        else:
            dH = dH_up + dH_partial
            dC = dC_partial + dH * self.O * tanh_backward_npdl(self.C)

        # Wc -- bc
        dL_dc_prev = dC * self.U * (1 - self.c_temp ** 2)
        self.dWc_h = H_prev.T.dot(dL_dc_prev)
        self.dWc_i = X.T.dot(dL_dc_prev)
        self.dbc = np.sum(dL_dc_prev, axis=0)

        # Wu -- bu
        dL_dU = dC * self.c_temp * self.U * (1 - self.U)
        self.dWu_h = H_prev.T.dot(dL_dU)
        self.dWu_i = X.T.dot(dL_dU)
        self.dbu = np.sum(dL_dU, axis=0)

        # Wf -- bf
        dL_dF = dC * c_prev * self.F * (1 - self.F)
        self.dWf_h = H_prev.T.dot(dL_dF)
        self.dWf_i = X.T.dot(dL_dF)
        self.dbf = np.sum(dL_dF, axis=0)

        # Wo -- bo
        dL_dO = dH * tanh_forward_npdl(self.C) * self.O * (1 - self.O)
        self.dWo_h = H_prev.T.dot(dL_dO)
        self.dWo_i = X.T.dot(dL_dO)
        self.dbo = np.sum(dL_dO, axis=0)

        # dH_prev
        dH_prev = dL_dc_prev.dot(self.Wc_h.T) + dL_dU.dot(self.Wu_h.T) \
                  + dL_dF.dot(self.Wf_h.T) + dL_dO.dot(self.Wo_h.T)

        # dC_prev
        dC_prev = dC * self.F

        # dX
        dX = dL_dc_prev.dot(self.Wc_i.T) + dL_dU.dot(self.Wu_i.T) \
             + dL_dF.dot(self.Wf_i.T) + dL_dO.dot(self.Wo_i.T)

        if mask is not None:
            dH_prev[mask] = dH_up[mask] + dH_partial[mask]
            dC_prev[mask] = dC_partial[mask]
            dX[mask] = np.zeros(self.input_size)

        return dX, dH_prev, dC_prev

    def get_gradients(self):
        return {'Wc_i': self.dWc_i,
                'Wu_i': self.dWu_i,
                'Wf_i': self.dWf_i,
                'Wo_i': self.dWo_i,
                'Wc_h': self.dWc_h,
                'Wu_h': self.dWu_h,
                'Wf_h': self.dWf_h,
                'Wo_h': self.dWo_h,
                'bc': self.dbc,
                'bu': self.dbu,
                'bf': self.dbf,
                'bo': self.dbo}
