import numpy as np
import math
from utils.activations import softmax_forward_npdl, softmax_backward_npdl


class SelfAttention:
    def __init__(self, d_model, d_k, d_v, weight_scale=1e-4, head_id=0):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.weight_scale = weight_scale
        self.head_id = str(head_id)

        self.WQ = self.init_weights(d_model, d_k, weight_scale)
        self.WK = self.init_weights(d_model, d_k, weight_scale)
        self.WV = self.init_weights(d_model, d_v, weight_scale)

        self.dWQ = np.zeros(self.WQ.shape)
        self.dWK = np.zeros(self.WK.shape)
        self.dWV = np.zeros(self.WV.shape)

        self.reg = 0.0
        self.cache = None

    def init_weights(self, dim_input, dim_output, weight_scale=1e-4):
        if weight_scale is not None:
            return np.random.normal(loc=0.0, scale=weight_scale, size=(dim_input, dim_output))
        else:
            return np.random.normal(loc=0.0, scale=math.sqrt(2.0 / (dim_input + dim_output)),
                                    size=(dim_input, dim_output))

    def forward_npdl(self, Q, K, V, **kwargs):
        """Propagation avant dans SelfAttention.

        Arguments:
            Q {ndarray} -- "Queries" de l'opération d'attention.
                           Shape (N, L, d_model)
            K {ndarray} -- "Keys" de l'opération d'attention.
                           Shape (N, L, d_model)
            V {ndarray} -- "Values" de l'opération d'attention.
                           Shape (N, L, d_model)

        Returns:
            ndarray -- Résultat de l'opération d'attention.
                       Shape (N, L, d_v)
        """

        Qi = Q.dot(self.WQ)
        Ki = K.dot(self.WK)
        Vi = V.dot(self.WV)

        output_mask = kwargs.get('output_mask', np.ones(Vi.shape[:-1] + (1,)))
        rhs_masked_timesteps = kwargs.get('rhs_mask', (np.array([], dtype=int), np.array([], dtype=int)))

        Z = np.matmul(Qi, Ki.transpose(0, 2, 1)) / np.sqrt(self.d_k)

        # Attention mask
        attention_masked_timesteps = np.where(output_mask == 0)
        Z[attention_masked_timesteps[0], :, attention_masked_timesteps[1]] = -np.exp(42)
        # rhs mask: utilisé dans le décodeur, masque toutes les positions à la droite d'une position i
        Z[:, rhs_masked_timesteps[0], rhs_masked_timesteps[1]] = -np.exp(42)

        S = softmax_forward_npdl(Z)
        S_masked = np.copy(S)
        S_masked[attention_masked_timesteps[0], attention_masked_timesteps[1], :] = 0

        self.cache = Q, K, V, Qi, Ki, Vi, S, S_masked, attention_masked_timesteps

        return np.matmul(S_masked, Vi)

    def backward_npdl(self, dA, **kwargs):
        """Rétropropagation dans SelfAttention.

        Arguments:
            dA {ndarray} -- Gradients de la loss par rapport à la sortie de la couche.
                            Shape (N, L, d_v)

        Returns:
            tuple -- Tuple contenant les gradients partiels de la loss par rapport 
                     à Q, K et V.
        """

        N = dA.shape[0]
        L = dA.shape[1]
        Q, K, V, Qi, Ki, Vi, S, S_masked, attention_masked_timesteps = self.cache

        M = (N * L) - len(attention_masked_timesteps[0])
        if M == 0:
            M = 1
            N = 0

        dS = np.matmul(dA, Vi.transpose(0, 2, 1))  # (N,L,L)
        dS[attention_masked_timesteps[0], attention_masked_timesteps[1], :] = 0

        dZ = softmax_backward_npdl(S, dS)  # (N,L,L)

        dQi = np.matmul(dZ, (Ki / np.sqrt(self.d_k)))  # (N,L,d_k)
        dKi = np.matmul(dZ, (Qi / np.sqrt(self.d_k)))  # (N,L,d_k)
        dVi = np.matmul(S_masked, dA)  # (N,L,d_v)

        dQ_partial = dQi.dot(self.WQ.T)
        dK_partial = dKi.dot(self.WK.T)
        dV_partial = dVi.dot(self.WV.T)

        self.dWQ = np.einsum('ijk,ijm->km', Q, dQi) * N / M + self.reg * self.WQ
        self.dWK = np.einsum('ijk,ijm->km', K, dKi) * N / M + self.reg * self.WK
        self.dWV = np.einsum('ijk,ijm->km', V, dVi) * N / M + self.reg * self.WV

        return dQ_partial, dK_partial, dV_partial

    def get_params(self, save=False):
        return {'WQ-' + self.head_id: self.WQ,
                'WK-' + self.head_id: self.WK,
                'WV-' + self.head_id: self.WV}

    def set_param(self, param, value):
        if param == 'WQ':
            assert self.WQ.shape == value.shape
            self.WQ = value
        elif param == 'WK':
            assert self.WK.shape == value.shape
            self.WK = value
        elif param == 'WV':
            assert self.WV.shape == value.shape
            self.WV = value
        elif param == 'reg':
            self.reg = value
        else:
            raise Exception(param + " is not a valid parameter for SelfAttention layers")

    def get_gradients(self):
        return {'WQ-' + self.head_id: self.dWQ,
                'WK-' + self.head_id: self.dWK,
                'WV-' + self.head_id: self.dWV}

    def reset(self):
        self.__init__(self.d_model,
                      self.d_k,
                      self.d_v,
                      self.weight_scale,
                      self.head_id)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads=0, weight_scale=1e-4):
        if num_heads == 0:
            num_heads = d_model
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_model = d_model
        self.attention = []
        for i in range(num_heads):
            self.attention.append(
                SelfAttention(d_model, d_model // num_heads, d_model // num_heads, weight_scale=weight_scale,
                              head_id=i))

        if weight_scale is not None:
            # Initialisation avec une distribution normale avec écart type = weight_scale
            self.W = np.random.normal(loc=0.0, scale=weight_scale, size=(d_model, d_model))
        else:
            # Initialisation 'Xavier' avec une distribution normale
            self.W = np.random.normal(loc=0.0, scale=math.sqrt(1.0 / d_model), size=(d_model, d_model))

        self.dW = 0
        self.reg = 0.0
        self.cache = None

    def forward_npdl(self, Q, K, V, **kwargs):
        """Propagation avant dans MultiHeadAttention.

        Arguments:
            Q {ndarray} -- "Queries" de l'opération d'attention.
                           Shape (N, L, d_model)
            K {ndarray} -- "Keys" de l'opération d'attention.
                           Shape (N, L, d_model)
            V {ndarray} -- "Values" de l'opération d'attention.
                           Shape (N, L, d_model)

        Returns:
            ndarray -- Résultat de l'opération d'attention multiple.
                       Shape (N, L, d_model)
        """

        assert Q.shape[2] == self.d_model
        assert K.shape[2] == self.d_model
        assert V.shape[2] == self.d_model
        outputs = []
        for i in range(self.h):
            outputs.append(self.attention[i].forward_npdl(Q, K, V, **kwargs))

        self.cache = np.concatenate(outputs, axis=2)

        return np.dot(self.cache, self.W)

    def backward_npdl(self, dOutput, **kwargs):
        """Rétropropagation dans MultiHeadAttention.

        Arguments:
            dA {ndarray} -- Gradients de la loss par rapport à la sortie de la couche.
                            Shape (N, L, d_model)

        Returns:
            ndarray -- Gradients de l'opération d'attention multiple.
                       Shape (N, L, d_model)
        """

        N = dOutput.shape[0]
        L = dOutput.shape[1]

        M = (N * L) - len(np.where((self.cache == 0).all(axis=2))[0])
        if M == 0:
            M = 1
            N = 0

        self.dW = np.einsum('ijm,ijk->mk', self.cache, dOutput) * N / M + self.reg * self.W
        dout = np.split(np.dot(dOutput, self.W.T), self.h, axis=2)

        dX = []
        for i in range(self.h):
            dX.append(self.attention[i].backward_npdl(dout[i], **kwargs))

        return np.sum(np.array(dX), axis=0)

    def get_params(self, save=False):
        params = {'W': self.W}
        for head in self.attention:
            head_params = head.get_params()
            params = {**params, **head_params}

        return params

    def set_param(self, param, value):
        if param == 'reg':
            for head in self.attention:
                head.set_param(param, value)
        elif param == 'W':
            self.W = value
        else:
            name, head_id = param.split('-')
            self.attention[int(head_id)].set_param(name, value)

    def get_gradients(self):
        gradients = {'W': self.dW}
        for head in self.attention:
            head_gradients = head.get_gradients()
            gradients = {**gradients, **head_gradients}

        return gradients

    def reset(self):
        self.__init__(self.d_model,
                      num_heads=self.h,
                      weight_scale=self.weight_scale)
