import numpy as np


def solver_npdl(X_train, y_train, X_val, y_val, reg, optimizer, lr_decay=1.0, num_iter=100, batch_size=200, verbose=True):
    """Boucle d'entraînement générale d'un modèle. La méthode de descente de gradient varie
       en fonction de l'optimizer.

    Arguments:
        X_train {ndarray} -- Données d'entraînement.
                             Shape (N, D)
        y_train {ndarray} -- Labels d'entraînement.
                             Shape (N, )
        X_val {ndarray} -- Données de validation.
                           Shape (N_val, D)
        y_val {ndarray} -- Labels de validation
                           Shape (N_val, )
        reg {float} -- Terme de régularisation
        optimizer {Optimizer} -- Objet permettant d'optimizer les paramètres du modèle.

    Keyword Arguments:
        lr_decay {float} -- Paramètre de contrôle pour la diminution du taux d'apprentissage.
                            (default: {1.0})
        num_iter {int} -- Nombre d'itérations d'entraînement (default: {100})
        batch_size {int} -- Nombre de données d'entraînement passées au modèle par itération.
                            (default: {200})
        verbose {bool} -- (default: {True})

    Returns:
        tuple -- Tuple contenant l'historique de la loss et de l'accuracy d'entraînement et de
                 validation
    """

    model = optimizer.model

    N = len(y_train)
    iterations_per_epoch = max(N / batch_size, 1)

    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for it in range(num_iter):
        batch_indices = np.random.choice(N, batch_size)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        output = model.forward_npdl(X_batch)
        loss, dScores, _ = model.calculate_loss(output, y_batch, reg)
        loss_history.append(loss)
        model.backward_npdl(dScores)

        optimizer.step()

        if verbose and it % 500 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iter, loss))

        # Après chaque epoch, on évalue l'accuracy d'entraînement et de
        # validation, et on diminue le learning rate (lr_decay = 1.0
        # veut dire que le learning rate reste fixe)
        if it % iterations_per_epoch == 0:
            train_accuracy = (model.predict(X_train) == y_train).mean()
            val_accuracy = (model.predict(X_val) == y_val).mean()
            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(val_accuracy)

            optimizer.lr *= lr_decay

    return loss_history, train_accuracy_history, val_accuracy_history


def epoch_solver_npdl(X_train, y_train, X_val, y_val, reg, optimizer, lr_decay=1.0, epochs=10, batch_size=100, verbose=True):
    """Boucle d'entraînement générale d'un modèle par epochs. La méthode de descente de gradient varie
       en fonction de l'optimizer.

    Arguments:
        X_train {ndarray} -- Données d'entraînement.
                             Shape (N, D)
        y_train {ndarray} -- Labels d'entraînement.
                             Shape (N, )
        X_val {ndarray} -- Données de validation.
                           Shape (N_val, D)
        y_val {ndarray} -- Labels de validation
                           Shape (N_val, )
        reg {float} -- Terme de régularisation
        optimizer {Optimizer} -- Objet permettant d'optimizer les paramètres du modèle.

    Keyword Arguments:
        lr_decay {float} -- Paramètre de contrôle pour la diminution du taux d'apprentissage.
                            (default: {1.0})
        epochs {int} -- Nombre d'epochs d'entraînement (default: {10})
        batch_size {int} -- Nombre de données d'entraînement passées au modèle par itération.
                            (default: {200})
        verbose {bool} -- (default: {True})

    Returns:
        tuple -- Tuple contenant l'historique de la loss et de l'accuracy d'entraînement et de
                 validation
    """

    model = optimizer.model

    N = len(y_train)

    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for epoch in range(epochs):
        batches = sample_npdl(X_train, y_train, batch_size)

        i = 0
        epoch_loss = 0
        for X_batch, y_batch in batches:
            output = model.forward_npdl(X_batch)
            loss, dScores, _ = model.calculate_loss(output, y_batch, reg)
            loss_history.append(loss)
            epoch_loss += loss
            model.backward_npdl(dScores)

            optimizer.step()

            i += 1
            if verbose and i % 20 == 0:
                print('(batch %d / %d) loss: %f' % (i, len(batches), loss))

        optimizer.lr *= lr_decay

        train_accuracy = check_accuracy(X_train, y_train, batch_size, model)
        val_accuracy = check_accuracy(X_val, y_val, batch_size, model)
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)

        if verbose:
            print('(epoch %d / %d) loss: %f, train_acc: %f, val_acc: %f' %
                  (epoch + 1, epochs, epoch_loss / i, train_accuracy, val_accuracy))

    return loss_history, train_accuracy_history, val_accuracy_history


def epoch_solver_seq2seq(X_train, Y_train, t_train, reg, optimizer, lr_decay=1.0, epochs=10, batch_size=100,
                         verbose=True):
    """Boucle d'entraînement par epochs spécifique aux modèles seq2seq. La méthode de descente de gradient varie
       en fonction de l'optimizer. Contrairement à epoch_solver, le calcul de l'accuracy n'a pas été implémenté.
       Une métrique comme le BLEU score serait plus appropriée.

    Arguments:
        X_train {ndarray} -- Données d'entraînement.
                             Shape (N, L, D)
        Y_train {ndarray} -- Targets d'entraînement.
                             Shape (N, L, D)
        t_train {ndarray} -- Labels d'entraînement.
                             Shape (N, L)
        reg {float} -- Terme de régularisation
        optimizer {Optimizer} -- Objet permettant d'optimizer les paramètres du modèle.

    Keyword Arguments:
        lr_decay {float} -- Paramètre de contrôle pour la diminution du taux d'apprentissage.
                            (default: {1.0})
        epochs {int} -- Nombre d'epochs d'entraînement (default: {10})
        batch_size {int} -- Nombre de données d'entraînement passées au modèle par itération.
                            (default: {200})
        verbose {bool} -- (default: {True})
    """

    model = optimizer.model

    N = len(t_train)

    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for epoch in range(epochs):
        batches = sample_seq2seq(X_train, Y_train, t_train, batch_size)

        i = 0
        epoch_loss = 0
        for X_batch, Y_batch, t_batch in batches:
            output = model.forward_npdl(X_batch, Y_batch)
            loss, dScores, _ = model.calculate_loss(output, t_batch, reg)
            loss_history.append(loss)
            epoch_loss += loss
            model.backward_npdl(dScores)

            optimizer.step()

            i += 1
            if verbose and i % 20 == 0:
                print('(batch %d / %d) loss: %f' % (i, len(batches), loss))

        optimizer.lr *= lr_decay

        if verbose:
            print('(epoch %d / %d) loss: %f' %
                  (epoch + 1, epochs, epoch_loss / i))

    return loss_history, train_accuracy_history, val_accuracy_history


def check_accuracy(X, y, batch_size, model):
    batches = sample_npdl(X, y, batch_size)

    accuracy = np.array([])

    for X_batch, y_batch in batches:
        accuracy = np.append(accuracy, (model.predict(X_batch) == y_batch))

    return accuracy.mean()


def sample_npdl(X, y, batch_size):
    """Permet de générer des batch d'entraînement pour une epoch.

    Arguments:
        X {ndarray} -- Données d'entraînement.
                       Shape (N, D)
        y {ndarray} -- Labels d'entraînement.
                       Shape (N, )
        batch_size {int} -- Taille d'une batch d'entraînement.

    Returns:
        list -- Liste contenant des tuples (X_batch, y_batch).
    """

    N = len(y)
    mod = N % batch_size

    if N <= batch_size:
        return [(X, y)]

    shuffle_indices = np.random.permutation(N)

    X_shuffle = X[shuffle_indices]
    y_shuffle = y[shuffle_indices]

    X_split = np.split(X_shuffle[:N - mod], int(N) // int(batch_size))
    y_split = np.split(y_shuffle[:N - mod], int(N) // int(batch_size))

    if N - mod < N:
        X_split.append(X_shuffle[N - mod:])
        y_split.append(y_shuffle[N - mod:])

    return [(X_split[i], y_split[i]) for i in range(len(X_split))]


def sample_seq2seq(X, Y, t, batch_size):
    """Permet de générer des batch d'entraînement pour une epoch.

    Arguments:
        X {ndarray} -- Données d'entraînement.
                       Shape (N, L, D)
        Y {ndarray} -- Targets d'entraînement.
                       Shape (N, L, D)
        t {ndarray} -- Labels d'entraînement.
                       Shape (N, L)
        batch_size {int} -- Taille d'une batch d'entraînement.

    Returns:
        list -- Liste contenant des tuples (X_batch, Y_batch, t_batch).
    """

    N = len(t)
    mod = N % batch_size

    if N <= batch_size:
        return [(X, Y, t)]

    shuffle_indices = np.random.permutation(N)

    X_shuffle = X[shuffle_indices]
    Y_shuffle = Y[shuffle_indices]
    t_shuffle = t[shuffle_indices]

    X_split = np.split(X_shuffle[:N - mod], int(N) // int(batch_size))
    Y_split = np.split(Y_shuffle[:N - mod], int(N) // int(batch_size))
    t_split = np.split(t_shuffle[:N - mod], int(N) // int(batch_size))

    if N - mod < N:
        X_split.append(X_shuffle[N - mod:])
        Y_split.append(Y_shuffle[N - mod:])
        t_split.append(t_shuffle[N - mod:])

    return [(X_split[i], Y_split[i], t_split[i]) for i in range(len(X_split))]


class SGD:
    def __init__(self, lr, model, momentum=0):
        self.initial_lr = lr

        self.lr = lr
        self.model = model
        self.momentum = momentum
        self.velocity = {}

        params = self.model.parameters()
        for layer_name, layer_params in params.items():
            tmp = {}

            for param_name, param in layer_params.items():
                tmp[param_name] = np.zeros(param.shape)

            self.velocity[layer_name] = tmp

    def step(self):
        """Applique l'algorithme de descente de gradient par batch
           à l'ensemble des paramètres du modèle.
        """

        params = self.model.parameters()
        grads = self.model.gradients()

        for layer_name, layer_params in params.items():
            for param_name, param in layer_params.items():
                self.velocity[layer_name][param_name] *= self.momentum
                self.velocity[layer_name][param_name] += self.lr * grads[layer_name][param_name]
                param -= self.velocity[layer_name][param_name]

    def set_lr(self, lr):
        self.initial_lr = lr
        self.lr = lr

    def reset(self):
        self.__init__(self.initial_lr,
                      self.model,
                      momentum=self.momentum)


class Adam:
    def __init__(self, lr, model, beta1=0.9, beta2=0.999, eps=1e-8):
        self.initial_lr = lr

        self.lr = lr
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 1

        params = self.model.parameters()
        for layer_name, layer_params in params.items():
            tmp_v = {}
            tmp_m = {}

            for param_name, param in layer_params.items():
                tmp_v[param_name] = np.zeros(param.shape)
                tmp_m[param_name] = np.zeros(param.shape)

            self.v[layer_name] = tmp_v
            self.m[layer_name] = tmp_m

    def step(self):
        """Applique l'algorithme de descente de gradient Adam
           à l'ensemble des paramètres du modèle.
        """

        params = self.model.parameters()
        grads = self.model.gradients()

        for layer_name, layer_params in params.items():
            for param_name, param in layer_params.items():
                self.m[layer_name][param_name] = self.beta1 * self.m[layer_name][param_name] + \
                                                 (1 - self.beta1) * \
                                                 grads[layer_name][param_name]
                self.v[layer_name][param_name] = self.beta2 * self.v[layer_name][param_name] + \
                                                 (1 - self.beta2) * \
                                                 (grads[layer_name][param_name] ** 2)

                m_corrected = self.m[layer_name][param_name] / (1 - self.beta1 ** self.t)
                v_corrected = self.v[layer_name][param_name] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.eps)

        self.t += 1

    def set_lr(self, lr):
        self.initial_lr = lr
        self.lr = lr

    def reset(self):
        self.__init__(self.initial_lr,
                      self.model,
                      beta1=self.beta1,
                      beta2=self.beta2,
                      eps=self.eps)
