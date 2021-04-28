# Ce code n'est pas utilisé dans la librairie. Il sert plutot d'exemple plus intuitif que
# le code cython équivalent.

import numpy as np


def get_im2col_indices(X_shape, Fheight, Fwidth, padding=(0, 0),
                       stride=(1, 1)):
    """Retourne les index sur trois dimensions (channel, lignes, colonnes)
       pour passer d'une entrée 3D régulière à une entrée matricisée, ou
       vice versa. Les mêmes index sont appliqués sur tous les éléments de X.

    Arguments:
        X_shape {tuple} -- Dimensions de X. (N, channel, height, width)
        Fheight {int} -- Hauteur du filtre.
        Fwidth {int} -- Largeur du filtre.

    Keyword Arguments:
        padding {tuple} -- Valeur du padding vertical et horizontal.
                           (default: {(1, 1)})
        stride {tuple} --  Valeur de la stride verticale et horizontale.
                           (default: {(1, 1)})

    Returns:
        tuple -- Tuple contenant les indices des trois dimensions pour
                 la "matricisation/dématricisation".
    """

    pad_height, pad_width = padding
    stride_height, stride_width = stride

    # Validation des dimensions avec le padding et les strides
    N, channel, height, width = X_shape
    assert (height + 2 * pad_height - Fheight) % stride_height == 0
    assert (width + 2 * pad_width - Fwidth) % stride_width == 0

    out_height = (height + 2 * pad_height - Fheight) // stride_height + 1
    out_width = (width + 2 * pad_width - Fwidth) // stride_width + 1

    # Élément de X à la rangée i, colonne j: xij

    ##################################################################
    # Index i: les index i indiquent, pour chaque élément de X_col   #
    # (X matricisé), l'index rangée de la valeur originale dans X.   #
    # Shape I: (channel * Fheight * Fwidth, out_height * out_width)  #
    ##################################################################

    # I_intra: Représente les index rangée "intra-colonne"
    I_intra = np.repeat(np.arange(Fheight, dtype=np.int), Fwidth)
    I_intra = np.tile(I_intra, channel)
    # I_stride: Ajoute l'effet de la stride verticale aux index
    # intra-colonne.
    I_stride = stride_height * np.repeat(np.arange(out_height, dtype=np.int), out_width)
    I = I_intra.reshape(-1, 1) + I_stride.reshape(1, -1)

    ##################################################################
    # Index j: les index j indiquent, pour chaque élément de X_col   #
    # (X matricisé), l'index colonne de la valeur originale dans X.  #
    # Shape J: (channel * Fheight * Fwidth, out_height * out_width)  #
    ##################################################################

    # J_intra: Représente les index colonne "intra-colonne"
    J_intra = np.tile(np.arange(Fwidth, dtype=np.int), Fheight * channel)
    # J_stride: Ajoute l'effet de la stride horizontale aux index
    J_stride = stride_width * np.tile(np.arange(out_width, dtype=np.int), out_height)
    J = J_intra.reshape(-1, 1) + J_stride.reshape(1, -1)

    ##################################################################
    # Index k: les index k indiquent à quel channel original         #
    # chaque élément d'une colonne de X_col (X matricisé) appartient.#
    # Le pattern d'appartenance aux channels étant identique pour    #
    # chaque colonne de X_col, K sera un vecteur de shape            #
    # (channel * Fheight * Fwidth, 1).                               #
    ##################################################################
    K = np.repeat(np.arange(channel, dtype=np.int), Fheight * Fwidth).reshape(-1, 1)

    return (K, I, J)


def im2col_indices(X, Fheight, Fwidth, padding=0, stride=1):
    """Permet de "matriciser" une image.

    Arguments:
        X {ndarray} -- Ensemble des images à matriciser.
                       Shape (N, channel, height, width)
        Fheight {int} -- Hauteur du filtre.
        Fwidth {int} -- Largeur du filtre.

    Keyword Arguments:
        padding {int or tuple} -- Valeur du padding. Si de type tuple,
                                  le padding vertical et horizontal peut
                                  être différent. (default: {1})
        stride {int or tuple} --  Valeur du stride. Si de type tuple,
                                  le stride vertical et horizontal peut
                                  être différent. (default: {1})

    Returns:
        ndarray -- Image matricisée
    """

    if isinstance(padding, tuple):
        pad_height, pad_width = padding
    elif isinstance(padding, int):
        pad_height, pad_width = padding, padding

    if isinstance(stride, tuple):
        stride_height, stride_width = stride
    elif isinstance(stride, int):
        stride_height, stride_width = stride, stride

    channel = X.shape[1]

    X_padded = np.pad(X, ((0, 0), (0, 0), (pad_height, pad_height),
                      (pad_width, pad_width)), mode='constant')

    K, I, J = get_im2col_indices(X.shape, Fheight, Fwidth,
                                 (pad_height, pad_width),
                                 (stride_height, stride_width))

    X_col = X_padded[:, K, I, J]
    X_col = X_col.transpose(1, 2, 0).reshape(Fheight * Fwidth * channel, -1)

    return X_col


def col2im_indices(X_col, X_shape, Fheight=3, Fwidth=3, padding=0,
                   stride=1):
    """Inverse de im2col_indices. Permet d'obtenir une image à
       partir d'une image "matricisée".

    Arguments:
        X_col {ndarray} -- Image matricisée.
                           Shape (channel * Fheight * Fwidth,
                                  N * out_height * out_width)
        X_shape{tuple} -- Dimensions de X. (N, channel, height, width)

    Keyword Arguments:
        Fheight {int} -- Hauteur du filtre. (default: {3})
        Fwidth {int} -- Largeur du filtre. (default: {3})
        padding {int or tuple} -- Valeur du padding. Si de type tuple,
                                  le padding vertical et horizontal peut
                                  être différent. (default: {1})
        stride {int or tuple} --  Valeur du stride. Si de type tuple,
                                  le stride vertical et horizontal peut
                                  être différent. (default: {1})

    Returns:
        ndarray -- "Image" dématricisée (ou gradients
                   dématricisés).
    """

    if isinstance(padding, tuple):
        pad_height, pad_width = padding
    elif isinstance(padding, int):
        pad_height, pad_width = padding, padding

    if isinstance(stride, tuple):
        stride_height, stride_width = stride
    elif isinstance(stride, int):
        stride_height, stride_width = stride, stride

    N, channel, height, width = X_shape
    pad_height = height + 2 * pad_height
    pad_width = width + 2 * pad_width

    X_padded = np.zeros((N, channel, pad_height, pad_width), dtype=X_col.dtype)
    K, I, J = get_im2col_indices(X_shape, Fheight, Fwidth,
                                 (pad_height, pad_width),
                                 (stride_height, stride_width))

    X_col_reshaped = X_col.reshape(channel * Fheight * Fwidth, -1, N)
    X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)

    np.add.at(X_padded, (slice(None), K, I, J), X_col_reshaped)

    if pad_height > 0 and pad_width > 0:
        X = X_padded[:, :, pad_height:-pad_height, pad_width:-pad_width]
    elif pad_height > 0:
        X = X_padded[:, :, pad_height:-pad_height, :]
    elif pad_width > 0:
        X = X_padded[:, :, :, pad_width:-pad_width]
    else:
        X = X_padded

    return X
