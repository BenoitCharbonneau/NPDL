import numpy as np


def confidence(model, X):
    """Calcule le niveau de confiance d'une prédiction du modèle.

    Arguments:
        model {Model} -- Le modèle.
        X {ndarray} -- Donnée en entrée.

    Returns:
        float -- niveau de confiance ([0, 1]) pour la prédiction.
    """

    scores = model.forward_npdl(X, mode='test')
    _, _, output = model.calculate_loss(scores, [0], 0.0)
    return np.max(output)


def mean_attack_error(model, X, X_adv):
    """Calcule l'erreur L1 relative entre les cartes d'activation produites par
       l'image originale et l'image attaquée.

    Arguments:
        model {Model} -- Le modèle.
        X {ndarray} -- La ou les images originales. Shape (H, W) ou (N, H, W)
        X_adv {ndarray} -- La ou les images d'attaque. Shape (H, W) ou (N, H, W)

    Returns:
        dict -- Dictionnaire des erreurs de chaque couche du réseau.
    """

    N = len(X)

    X_data = model.forward_npdl(X, mode='test', all=True)
    X_adv_data = model.forward_npdl(X_adv, mode='test', all=True)

    mse_model = {}
    for name in X_data.keys():
        if X.shape[0] > 1:
            axis = tuple([i + 1 for i in range(len(X_data[name].shape) - 1)])
            reshape = (N,) + (1,) * (len(X_data[name].shape) - 1)
            X_norm = (X_data[name] - np.min(X_data[name], axis=axis).reshape(reshape)) / \
                     (np.max(X_data[name], axis=axis) - np.min(X_data[name], axis=axis)).reshape(reshape)
            X_adv_norm = (X_adv_data[name] - np.min(X_data[name], axis=axis).reshape(reshape)) / \
                         (np.max(X_data[name], axis=axis) - np.min(X_data[name], axis=axis)).reshape(reshape)
            mse = np.mean(np.mean(np.abs(X_norm - X_adv_norm), axis=axis))
        else:
            X_norm = (X_data[name] - np.min(X_data[name])) / (np.max(X_data[name]) - np.min(X_data[name]))
            X_adv_norm = (X_adv_data[name] - np.min(X_data[name])) / (np.max(X_data[name]) - np.min(X_data[name]))
            mse = np.mean(np.abs(X_norm - X_adv_norm))

            if name == "inputs":
                print("inputs")
            else:
                print(type(model.layers[name]).__name__)

            print(mse)

        mse_model[name] = mse

    return mse_model


def print_mse(mse_model, model):
    for name, mse in mse_model.items():
        if name == "inputs":
            print("inputs")
        else:
            print(type(model.layers[name]).__name__)

        print(mse)


def fgsm(model, X, y, eps=1):
    """Méthode d'attaque adverse FGSM. La logique derrière ce type
       d'attaque est de générer un delta dans lequel les valeurs 
       correspondent au -np.sign du gradient de la perte en fonction de
       l'entrée. Note: la perte n'est pas calculée avec la classe prédite
       par le modèle, mais plutôt avec la classe résultante désirée, ce qui
       permet de modifier l'image d'origine dans une "direction" qui
       influencera la prédiction dans le sens voulu.

    Arguments:
        model {Model} -- Le modèle.
        X {ndarray} -- L'image originale en entrée. Shape (1, H, W)
        y {int} -- Index de la classe visée.

    Keyword Arguments:
        eps {int} -- facteur multiplicatif de la perturbation (default: {1})

    Returns:
        tuple -- L'image perturbée et la perturbation.
    """

    scores = model.forward_npdl(X, mode='test')
    _, gradient, _ = model.calculate_loss(scores, [y], 0.0)
    input_gradient = model.backward_npdl(gradient, mode='test')
    perturb = eps * -np.sign(input_gradient)
    X_adv = np.clip(X + perturb, -128.0, 127.0)
    return X_adv, perturb


def jsma(model, X, scores, target, C, theta=10, maximum_distortion=128, max_modification_ratio=0.5):
    """Jacobian saliency map attack. Cette attaque "whitebox - targetted" consiste à modifier les
       caractéristiques de l'image dans une direction obtenue avec une fonction de salience
       appliquée sur le gradient des scores de chaque classe par rapport à chaque 
       caractéristique de l'image en entrée (d'où la jacobienne), ce qui permet de modifier seulement
       les caractéristiques qui affectent le plus la perte.

    Arguments:
        model {Model} -- Le modèle.
        X {ndarray} -- Image en entrée. Shape (1, channels, H, W)
        scores {ndarray} -- Scores initiaux. Shape (1, C)
        target {int} -- Classes visée.
        C {int} -- Nombre de classes.

    Keyword Arguments:
        theta {int} -- Modification appliquée à l'image lors d'une itération (default: {10})
        maximum_distortion {int} -- Valeur maximale (et - minimale) pour chaque canal (default: {128})
        max_modification_ratio {float} -- Pourcentage des caractéristiques maximal pouvant être
                                          modifiées (default: {0.5})

    Returns:
        ndarray -- Image perturbée. Shape (1, channels, H, W)
    """

    pred = model.predict(X)
    X_adv = np.copy(X)

    nb_iter = int(max_modification_ratio * X.shape[1] * X.shape[2] * X.shape[3])

    i = 0
    while pred != target and i < nb_iter:
        jacobian_4d = get_jacobian(model, C)
        jacobian = jacobian_4d.reshape((C, -1))

        pred_index = np.argmax(scores)

        dTarget = jacobian[target]
        dOthers = np.sum(jacobian, axis=0) - dTarget

        # Call saliency
        s = saliency_map(X_adv, dTarget, dOthers, maximum_distortion)

        # modify pixel
        target_pixel = np.argmax(s)
        target_pixel_2 = np.argpartition(s, -2)[-2]

        X_adv_r = X_adv.reshape(-1)
        X_adv_r[target_pixel] = max(min(X_adv_r[target_pixel] + theta, maximum_distortion), -maximum_distortion)
        X_adv_r[target_pixel_2] = max(min(X_adv_r[target_pixel_2] + theta, maximum_distortion), -maximum_distortion)

        # next iter
        X_adv = X_adv_r.reshape(X_adv.shape)
        pred = model.predict(X_adv)
        i += 1

    return X_adv


def get_jacobian(model, C):
    """Génère la matrice jacobienne des gradients des scores de chaque classe par rapport
       à chaque caractéristique de l'entrée.

    Arguments:
        model {Model} -- Le modèle.
        C {int} -- Nombre de classes.

    Returns:
        ndarray -- Jacobienne. Shape (1, C, channels, H, W)
    """

    for i in range(C):
        one_hot = np.zeros(C)
        one_hot[i] = 1

        if i == 0:
            jacobian = model.backward_npdl(one_hot, mode='test')
        else:
            jacobian = np.concatenate((jacobian, model.backward_npdl(one_hot, mode='test')), axis=0)

    return jacobian


def saliency_map(X, dTarget, dOthers, maximum_distortion):
    """Fonction de salience appliquée à la jacobienne. Pour une caractéristique
       i et une classe y, la fonction retourne 0 si dTarget < 0 ou si la somme
       des gradients pour les autres classes (dOther) est > 0, et retourne 
       np.abs(dOther) * dTarget sinon.

    Arguments:
        X {ndarray} -- L'image en entrée.
        dTarget {ndarray} -- Gradients de la classe target en fonction de l'entrée.
        dOthers {ndarray} -- Somme des gradients des autres classes en fonction de l'entrée (vecteur).
        maximum_distortion {int} -- Valeur maximale (et - minimale) pour chaque canal

    Returns:
        ndarray -- Scores générés par la fonction de salience.
    """

    X_r = X.reshape(-1)

    max_reached_filter = (X_r < maximum_distortion).astype(int)
    min_reached_filter = (X_r > -maximum_distortion).astype(int)

    dTarget_filter = (dTarget >= 0).astype(int)
    dOthers_filter = (dOthers <= 0).astype(int)

    return max_reached_filter * min_reached_filter * dTarget_filter * dOthers_filter * (dTarget * np.abs(dOthers))


def softmax(scores):
    stable_scores = scores - np.max(scores, axis=1)[:, None]
    exp_scores = np.exp(stable_scores)
    softmax_output = exp_scores / np.sum(exp_scores, axis=1)[:, None]
    return softmax_output


def simba(model, X, original_predictions, original_target, eps=2):
    """Simple blackbox attack. Cette attaque "blackbox - untargeted"
       consiste à aléatoirement modifier la valeur de pixels dans la 
       direction qui influence négativement le score de la classe prédite
       originale jusqu'à ce que le modèle prédise une autre classe.

    Arguments:
        model {Model} -- Le modèle.
        X {ndarray} -- Donnée originale. Shape (C, H, W)
        original_predictions {ndarray} -- Prédictions originales. Shape (nb_classes,)
        original_target {int} -- Index de la classe prédite originale.

    Keyword Arguments:
        eps {int} -- Facteur de modification d'une feature (default: {2})

    Raises:
        Exception: Survient lorsque l'attaque ne réussi pas à modifier la prédiction
                   du modèle.

    Returns:
        ndarray -- Perturbation à appliquer sur l'image originale. Shape (C, H, W)
    """

    C, H, W = X.shape

    delta = np.zeros(X.shape)
    picked = []

    p = np.copy(original_predictions)

    while original_target == np.argmax(p):
        if len(picked) >= C * H * W:
            raise Exception("No solution found")

        ix = (np.random.randint(0, C), np.random.randint(0, H), np.random.randint(0, W))
        while ix in picked:  # can loop infinitely. if its the case, just relaunch
            ix = (np.random.randint(0, C), np.random.randint(0, H), np.random.randint(0, W))

        picked.append(ix)

        for alpha in (eps, -eps):
            X_prime = np.copy(X) + delta
            X_prime[ix] += alpha
            p_prime = softmax(model.forward_npdl(X_prime[np.newaxis, :, :, :], mode='test')).reshape(-1, )

            if p_prime[original_target] < p[original_target]:
                delta[ix] += alpha
                p = p_prime
                break

    return delta
