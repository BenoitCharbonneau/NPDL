import numpy as np

def convolution_naive_npdl(x, w, b, conv_param, verbose=0):
    """
    Version naive de la propagation avant d'une convolution.  

    Le tenseur d'entrée x est une batch comprenant N images 2D de taille WxH, et 
    chacune ayant C canaux. Par exemple, si x est une batch 5 images couleur de 
    CIFAR10, alors x serait de taille 
    
    5 x 3 x 32 x 32
    
    x est donc convoluté avec F filtres dont les poids sont contenus dans "w". 
    Par exemple, si w contient 7 filtre de taille 5x5 et que les images dans x ont
    3 canaux, alors w sera de taille
    
    7 x 3 x 5 x 5
    
    Entrée:
    - x: tenseur d'entrée (N, C, H, W)
    - w: tenseur de poids du filtre (F, C, HH, WW)
    - b: vecteur de biais de taille (F)
    - conv_param: Dictionnaire comprenant les paramètres suivants:
      - 'stride': décalage horizontal et vertical lors de l'opération de convolution
      - 'pad': nombre de colonnes à gauche et à droite ainsi que de lignes en haut
               et en bas lors d'une opération de "zero padding".  Exemple, si pad = 1
               on ajoute un colonne de zéros à gauche et à droite et une ligne de
               zéros en haut et en bas.

    Retour:
    - out: tenseur convolué et taille (N, F, H', W') où H' et W' sont données par
           l'opération suivantes
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x_pad, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape

    pad = conv_param['pad']
    stride = conv_param['stride']

    #############################################################################
    # TODO: Implémentez la propagation pour la couche de convolution.           #
    # Astuces: vous pouvez utiliser la fonction np.pad pour le remplissage.     #
    #############################################################################
    assert (H - FH + 2 * pad) % stride == 0
    assert (W - FW + 2 * pad) % stride == 0
    outH = np.uint32(1 + (H - FH + 2 * pad) / stride)
    outW = np.uint32(1 + (W - FW + 2 * pad) / stride)

    # create output tensor after convolution layer
    out = np.zeros((N, F, outH, outW))

    # padding all input data
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    # create w_row matrix
    w_row = w.reshape(F, C*FH*FW)                            #[F x C*FH*FW]

    # create x_col matrix with values that each neuron is connected to
    x_col = np.zeros((C*FH*FW, outH*outW))                   #[C*FH*FW x H'*W']
    for index in range(N):
        neuron = 0
        for i in range(0, H_pad-FH+1, stride):
            for j in range(0, W_pad-FW+1, stride):
                x_col[:, neuron] = x_pad[index, :, i:i+FH, j:j+FW].reshape(C*FH*FW)
                neuron += 1
        out[index] = (w_row.dot(x_col) + b.reshape(F, 1)).reshape(F, outH, outW)
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    cache = (x_pad, w, b, conv_param)

    return out, cache


def backward_convolution_naive_npdl(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.  (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x : (N, C, H, W)
    - dw: Gradient with respect to w : (F, C, HH, WW)
    - db: Gradient with respect to b : (F,)
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implémentez la rétropropagation pour la couche de convolution       #
    #############################################################################
    
    x_pad, w, b, conv_param = cache
    N, F, outH, outW = dout.shape
    N, C, Hpad, Wpad = x_pad.shape
    FH, FW = w.shape[2], w.shape[3]
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # initialize gradients
    dx = np.zeros((N, C, Hpad - 2 * pad, Wpad - 2 * pad))
    dw, db = np.zeros(w.shape), np.zeros(b.shape)

    # create w_row matrix
    w_row = w.reshape(F, C * FH * FW)  # [F x C*FH*FW]

    # create x_col matrix with values that each neuron is connected to
    x_col = np.zeros((C * FH * FW, outH * outW))  # [C*FH*FW x H'*W']
    for index in range(N):
        out_col = dout[index].reshape(F, outH * outW)  # [F x H'*W']
        w_out = w_row.T.dot(out_col)  # [C*FH*FW x H'*W']
        dx_cur = np.zeros((C, Hpad, Wpad))
        neuron = 0
        for i in range(0, Hpad - FH + 1, stride):
            for j in range(0, Wpad - FW + 1, stride):
                dx_cur[:, i:i + FH, j:j + FW] += w_out[:, neuron].reshape(C, FH, FW)
                x_col[:, neuron] = x_pad[index, :, i:i + FH, j:j + FW].reshape(C * FH * FW)
                neuron += 1
                
        if pad > 0:
            dx[index] = dx_cur[:, pad:-pad, pad:-pad]
        else:
            dx[index] = dx_cur[:, :, :]
            
        dw += out_col.dot(x_col.T).reshape(F, C, FH, FW)
        db += out_col.sum(axis=1)

    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    return dx, dw, db
