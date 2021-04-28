# Code adapté de projets académiques de la professeur Fei Fei Li et
# de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Première version rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin.
# Version finale rédigée par Benoit Charbonneau et Pierre-Luc Parent


import numpy as np


def check_gradient_sparse(loss_func, W, X, y, reg=0.0, num_checks=10, h=1e-6):
    """Compare les gradients du modèle à un gradient numérique sur un nombre
       fini de paramètres du modèle.

    Arguments:
        loss_func {function} -- Fonction de perte.
        W {ndarray} -- Poids du modèle. Shape (D, C)
        X {ndarray} -- Données de validation. Shape (N, D)
        y {ndarray} -- Labels de validation. Shape (N, )

    Keyword Arguments:
        reg {float} -- terme de régularisation (default: {0.0})
        num_checks {int} -- Nombre de paramètres pour lesquels la comparaison sera faite.
                            (default: {10})
        h {float} -- Taille de la modification apportée à un paramètre pour calculer le
                     gradient numérique. (default: {1e-6})
    """
    _, grad = loss_func(X, W, y, reg)

    for i in range(num_checks):
        index = tuple([np.random.randint(0, m) for m in W.shape])
        w_value = W[index]
        W[index] = w_value + h
        next_step_loss, _ = loss_func(X, W, y, reg)
        W[index] = w_value - h
        prev_step_loss, _ = loss_func(X, W, y, reg)
        W[index] = w_value

        grad_numerical = (next_step_loss - prev_step_loss) / (2 * h)
        grad_analytic = grad[index]
        error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f, analytic %f, relative error: %e' % (grad_numerical, grad_analytic, error))


def evaluate_numerical_gradient(X, y, model, layer_name, param_name, reg=0.0, h=1e-5, **kwargs):
    """Compare les gradients du modèle à un gradient numérique. Appliqué sur tous les
       paramètres du modèle.

    Arguments:
        loss_func {function} -- Fonction de perte.
        X {ndarray} -- Données de validation. Shape (N, D)
        y {ndarray} -- Labels de validation. Shape (N, )
        model {Model} -- Modèle sur lequel l'évaludation des gradients est faite.
        layer_name {string} -- Nom de la couche à évaluer.
        param_name {string} -- Nom du paramètre (W ou b) à évaluer.

    Keyword Arguments:
        reg {float} -- terme de régularisation (default: {0.0})
        h {float} -- Taille de la modification apportée à un paramètre pour calculer le
                     gradient numérique. (default: {1e-6})
    """

    params = model.parameters()[layer_name][param_name]

    grad = np.zeros_like(params)

    it = np.nditer(params, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        ix = it.multi_index

        param_value = params[ix]

        params[ix] = param_value + h
        next_step_scores = model.forward_npdl(X, **kwargs)
        next_step_loss, _, _ = model.calculate_loss(next_step_scores, y, reg)

        params[ix] = param_value - h
        prev_step_scores = model.forward_npdl(X, **kwargs)
        prev_step_loss, _, _ = model.calculate_loss(prev_step_scores, y, reg)
        params[ix] = param_value

        grad[ix] = (next_step_loss - prev_step_loss) / (2 * h)

        it.iternext()

    return grad
