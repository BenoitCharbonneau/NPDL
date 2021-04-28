import numpy as np


def ce_naive_fb_npdl(X, W, y, reg):
    """Implémentation naive qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) + une régularisation L2 et le gradient des poids. Utilise une 
       activation softmax en sortie.
       
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2
       
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemple d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """

    N = X.shape[0]
    C = W.shape[1]

    loss = 0
    dW = 0

    for i in range(N):
        scores = X[i].dot(W)
        stable_scores = scores - np.max(scores)
        exp_scores = np.exp(stable_scores)

        softmax_output = exp_scores / np.sum(exp_scores)

        target_output = softmax_output[y[i]]

        loss += -np.log(target_output)

        dScores = softmax_output - (range(C) == y[i])

        dW += X[i][:, None].dot(dScores[None, :])

    loss /= N
    loss += 0.5 * reg * np.sum(np.square(W))

    dW /= N
    dW += reg * W

    return loss, dW


def ce_fb_npdl(X, W, y, reg):
    """Implémentation vectorisée qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) et le gradient des poids. Utilise une activation softmax en sortie.
        
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2      
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemples d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    N = X.shape[0]
    C = W.shape[1]

    scores = X.dot(W)
    stable_scores = scores - np.max(scores, axis=1)[:, None]
    exp_scores = np.exp(stable_scores)
    softmax_output = exp_scores / np.sum(exp_scores, axis=1)[:, None]

    target_indices_mask = np.eye(C, dtype='bool')[y]
    target_outputs = softmax_output[target_indices_mask]

    loss = -np.sum(np.log(target_outputs)) / N + 0.5 * reg * np.sum(np.square(W))

    dScores = softmax_output - target_indices_mask.astype(int)
    dW = X.T.dot(dScores) / N + reg * W

    return loss, dW


def hinge_naive_fb_npdl(X, W, y, reg):
    """Implémentation naive qui calcule la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.
       
       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2


    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    num_train = X.shape[0]
    for i in range(num_train):
        XW = X[i].dot(W)
        prediction = np.argmax(XW)

        m = max(0, 1 + XW[prediction] - XW[y[i]])
        loss += m
        if m > 0:
            dW[:, prediction] += X[i]
            dW[:, y[i]] -= X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(np.square(W))

    dW /= num_train
    dW += reg * W

    return loss, dW


def hinge_fb_npdl(X, W, y, reg):
    """Implémentation vectorisée qui calcule la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.

       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!
       
    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2


    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    num_train = X.shape[0]
    train_shape = np.arange(0, num_train)
    XW = X.dot(W)
    predictions = np.max(XW, axis=1)
    targets = XW[train_shape, y]
    m = np.maximum(0, 1 + predictions - targets)

    loss = np.sum(m) / num_train
    loss += 0.5 * reg * np.sum(np.square(W))

    mask = np.zeros(XW.shape)
    predictions_index = np.argmax(XW, axis=1)
    mask[train_shape, predictions_index] += 1.0
    mask[train_shape, y] -= 1.0
    dW = X.T.dot(mask)
    dW /= num_train
    dW += reg * W

    return loss, dW
