import numpy as np


def positional_encoding(X, mask):
    """Encodage positionel pour l'entrée d'un Transformer.

    Arguments:
        X {ndarray} -- données en entrée. Shape (N, L, d_model)
        mask {ndarray} --  masque (0 padding). Shape (N, L, 1)

    Returns:
            ndarray -- Entrées encodées. Shape (N, L, d_model)
    """

    N, L, d_model = X.shape

    assert N == mask.shape[0]
    assert L == mask.shape[1]

    pos_ix = np.arange(L)
    dim_ix = np.arange(d_model)

    pos = np.repeat(pos_ix[:, np.newaxis], d_model, axis=1)
    dim = np.repeat(dim_ix[:, np.newaxis], L, axis=1).T

    sin_encoding = np.sin(pos / np.power(10000, 2 * dim / d_model))
    cos_encoding = np.cos(pos / np.power(10000, 2 * dim / d_model))
    encoding = (-((dim_ix % 2) - 1)) * sin_encoding + (dim_ix % 2) * cos_encoding

    return (np.repeat(encoding[np.newaxis, :, :], N, axis=0) + X) * mask
