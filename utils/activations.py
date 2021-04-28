import numpy as np


def relu_forward_npdl(Z):
    return np.maximum(0, Z)


def relu_backward_npdl(Z):
    return (Z > 0).astype(int)


def identity_forward_npdl(Z):
    return Z


def identity_backward_npdl(Z):
    return 1


def sigmoid_forward_npdl(Z):
    return  1 / (1 + np.exp(-Z))


def sigmoid_backward_npdl(Z):
    sig = sigmoid_forward_npdl(Z)
    return sig * (1 - sig)


def tanh_forward_npdl(Z):
    return np.tanh(Z)


def tanh_backward_npdl(Z):
    return 1 - (np.tanh(Z) ** 2)


def softmax_forward_npdl(Z):
    max_Z = np.max(Z, axis=2)
    stable_Z = Z - np.expand_dims(max_Z, max_Z.ndim)
    exp_Z = np.exp(stable_Z)
    sum_Z = np.sum(exp_Z, axis=2)

    return exp_Z / np.expand_dims(sum_Z, sum_Z.ndim)


def softmax_backward_npdl(S, dLdS):
    C = S.shape[2]

    S_j = np.repeat(S[:, :, np.newaxis, :], C, axis=2)
    S_i = S[:, :, :, np.newaxis]
    I = np.eye(C)[np.newaxis, :, :]

    delta = I - S_j
    jacobian = S_i * delta
    return np.einsum('ijkl,ijl->ijk',jacobian, dLdS)


def get_activation(activation):
    """Permet d'obtenir l'activation associée au paramètre activation
       et sa fonction de dérivation.

    Arguments:
        activation {str} -- Identifiant de l'activation demandée.

    Returns:
        dict -- Dictionnaire contenant l'activation et sa fonction de 
                dérivation.
    """

    if activation == 'identity':
        return {'forward': identity_forward_npdl, 'backward': identity_backward_npdl}
    elif activation == 'relu':
        return {'forward': relu_forward_npdl, 'backward': relu_backward_npdl}
    else:
        raise Exception("Not a valid activation function")
