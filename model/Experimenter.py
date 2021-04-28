import numpy as np
from model.Solver import epoch_solver_npdl
from visualization.utils import visualize_loss, visualize_accuracy


class Experimenter:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.results_cache = {}
        self.hyperparam_cache = {}

    def train_npdl(self, data, training_config={}, verbose=True, experiment_id=None):
        """Entraîne le modèle de l'experimenter (par l'entremise de l'optimizer) selon
           la configuration spécifiée.

        Arguments:
            data {dict} -- Dictionnaire contenant X_train, y_train, X_val, y_val.

        Keyword Arguments:
            training_config {dict} -- Configuration contenant les différents hyperparamètres
                                      d'entraînement. (default: {{}})
            verbose {bool} -- Option verbose (default: {True})
            experiment_id {str} -- Identificateur de l'expérience (default: {'default_exp'})

        Returns:
            tuple -- Résultats de l'entraînement (historique de perte, 
                       d'accuracy d'entraînement et de validation).
        """

        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']

        lr = training_config.get('lr', 1e-3)
        reg = training_config.get('reg', 0.0)
        batch_size = training_config.get('batch_size', 100)
        lr_decay = training_config.get('lr_decay', 1.0)
        epochs = training_config.get('epochs', 10)

        self.optimizer.set_lr(lr)

        results = epoch_solver_npdl(X_train, y_train, X_val, y_val, reg, self.optimizer, lr_decay=lr_decay,
                               epochs=epochs, batch_size=batch_size, verbose=verbose)

        if experiment_id is not None:
            self.results_cache[experiment_id] = results
            self.hyperparam_cache[experiment_id] = {
                'lr': lr,
                'reg': reg,
                'batch_size': batch_size,
                'lr_decay': lr_decay,
                'epochs': epochs
            }

        return results

    def hyperparam_grid_search(self, data, epochs=5, K=5, hyperparams={}, experiment_id='default_grid', verbose=True):
        """Recherche d'hyperparamètres par méthode "grid search".

        Arguments:
            data {dict} -- Dictionnaire contenant X, y.

        Keyword Arguments:
            epochs {int} -- Nombre d'epochs d'entraînement pour chaque ensemble
                            d'hyperparamètres (default: {5})
            hyperparams {dict} -- Ensemble des valeurs à tester pour 
                                  chaque hyperparamètre (default: {{}})
            experiment_id {str} -- Identificateur de l'expérience (default: {'default_exp'})
            verbose {bool} -- Option verbose (default: {True})

        Returns:
            tuple -- Tuple contenant la liste des meilleurs hyperparamètres et 
                     l'identificateur de la meilleure expérience.
        """

        X = data['X']
        y = data['y']

        lr_range = hyperparams.get('lr', [1e-2, 1e-3, 1e-4])
        reg_range = hyperparams.get('reg', [0.0, 0.1, 0.01])
        batch_size_range = hyperparams.get('batch_size', [50, 100, 200])
        lr_decay_range = hyperparams.get('lr_decay', [1.0, 0.95, 0.9])

        count = 0
        best_val_accuracy = -np.inf
        best_experiment_id = None
        best_hyperparams = {}

        for lr in lr_range:
            for reg in reg_range:
                for batch_size in batch_size_range:
                    for lr_decay in lr_decay_range:
                        count += 1

                        current_experiment_id = experiment_id + '-' + str(count)

                        # Le même modèle est réutilisé, on doit donc réinitialiser ses paramètres
                        self.optimizer.model.reset()
                        self.optimizer.reset()

                        avg_loss_history = []
                        avg_train_accuracy_history = []
                        avg_val_accuracy_history = []

                        for k in range(K):
                            loss_history, train_accuracy_history, val_accuracy_history = self.hyperparam_iteration(
                                X, y, lr, reg, batch_size, lr_decay, epochs, None)

                            avg_loss_history.append(loss_history)
                            avg_train_accuracy_history.append(train_accuracy_history)
                            avg_val_accuracy_history.append(val_accuracy_history)

                        avg_loss_history = [np.sum(i) / K for i in zip(*avg_loss_history)]
                        avg_train_accuracy_history = [np.sum(i) / K for i in zip(*avg_train_accuracy_history)]
                        avg_val_accuracy_history = [np.sum(i) / K for i in zip(*avg_val_accuracy_history)]

                        self.results_cache[current_experiment_id] = (avg_loss_history, 
                                                                     avg_train_accuracy_history, 
                                                                     avg_val_accuracy_history)
                        self.hyperparam_cache[current_experiment_id] = {
                            'lr': lr,
                            'reg': reg,
                            'batch_size': batch_size,
                            'lr_decay': lr_decay,
                            'epochs': epochs
                        }                                                                                    

                        if avg_val_accuracy_history[-1] > best_val_accuracy:
                            best_val_accuracy = avg_val_accuracy_history[-1]
                            best_experiment_id = current_experiment_id

                            best_hyperparams = {
                                'lr': lr,
                                'reg': reg,
                                'batch_size': batch_size,
                                'lr_decay': lr_decay
                            }

                        if verbose:
                            print('lr: ', lr, ' -- reg: ', reg, ' -- batch_size: ', batch_size, ' -- lr_decay: ',
                                  lr_decay)
                            print('train_accuracy: ', avg_train_accuracy_history[-1])
                            print('val_accuracy: ', avg_val_accuracy_history[-1])
                        
        return best_hyperparams, best_experiment_id

    def hyperparam_random_search(self, data, epochs=5, K=5, vals_per_hyperparam=3, hyperparams={},
                                 experiment_id='default_random', verbose=True):
        """Recherche d'hyperparamètres par méthode "random search".

        Arguments:
            data {dict} -- Dictionnaire contenant X, y.

        Keyword Arguments:
            epochs {int} -- Nombre d'epochs d'entraînement pour chaque ensemble
                            d'hyperparamètres (default: {5})
            vals_per_hyperparam {int} -- Nombre de valeurs aléatoires à tester par 
                                         hyperparamètre (default: {3})
            hyperparams {dict} -- Bornes maximales et minimales des valeurs à tester pour 
                                  chaque hyperparamètre (default: {{}})
            experiment_id {str} -- Identificateur de l'expérience (default: {'default_exp'})
            verbose {bool} -- Option verbose (default: {True})

        Returns:
            tuple -- Tuple contenant la liste des meilleurs hyperparamètres et 
                     l'identificateur de la meilleure expérience.
        """

        X = data['X']
        y = data['y']

        lr_min, lr_max = hyperparams.get('lr', (1e-2, 1e-4))
        reg_min, reg_max = hyperparams.get('reg', (0.0, 0.01))
        lr_decay_min, lr_decay_max = hyperparams.get('lr_decay', (1.0, 0.9))

        batch_size = hyperparams.get('batch_size', 100)

        count = 0
        best_val_accuracy = -np.inf
        best_experiment_id = None
        best_hyperparams = {}

        for i in range(vals_per_hyperparam ** 4):
            count += 1

            lr = np.random.uniform(lr_min, lr_max)
            reg = np.random.uniform(reg_min, reg_max)
            lr_decay = np.random.uniform(lr_decay_min, lr_decay_max)

            current_experiment_id = experiment_id + '-' + str(count)

            # Le même modèle est réutilisé, on doit donc réinitialiser ses paramètres
            self.optimizer.model.reset()
            self.optimizer.reset()

            avg_loss_history = []
            avg_train_accuracy_history = []
            avg_val_accuracy_history = []

            for k in range(K):
                loss_history, train_accuracy_history, val_accuracy_history = self.hyperparam_iteration(
                    X, y, lr, reg, batch_size, lr_decay, epochs, None)

                avg_loss_history.append(loss_history)
                avg_train_accuracy_history.append(train_accuracy_history)
                avg_val_accuracy_history.append(val_accuracy_history)

            avg_loss_history = [np.sum(i) / K for i in zip(*avg_loss_history)]
            avg_train_accuracy_history = [np.sum(i) / K for i in zip(*avg_train_accuracy_history)]
            avg_val_accuracy_history = [np.sum(i) / K for i in zip(*avg_val_accuracy_history)]

            self.results_cache[current_experiment_id] = (avg_loss_history, 
                                                         avg_train_accuracy_history, 
                                                         avg_val_accuracy_history)
            self.hyperparam_cache[current_experiment_id] = {
                'lr': lr,
                'reg': reg,
                'batch_size': batch_size,
                'lr_decay': lr_decay,
                'epochs': epochs
            }

            if avg_val_accuracy_history[-1] > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy_history[-1]
                best_experiment_id = current_experiment_id

                best_hyperparams = {
                    'lr': lr,
                    'reg': reg,
                    'batch_size': batch_size,
                    'lr_decay': lr_decay
                }
            
            if verbose:
                print('lr: ', lr, ' -- reg: ', reg, ' -- batch_size: ', batch_size, ' -- lr_decay: ', lr_decay)
                print('train_accuracy: ', avg_train_accuracy_history[-1])
                print('val_accuracy: ', avg_val_accuracy_history[-1])

        return best_hyperparams, best_experiment_id

    def hyperparam_iteration(self, X, y, lr, reg, batch_size, lr_decay, epochs, experiment_id):
        """Effectue l'entraînement du modèle avec l'ensemble d'hyperparamètres donné.

        Arguments:
            X {ndarray} -- Données utilisées pour générer les données d'entraînement
                           et de validation.
            y {ndarray} -- Labels utilisés pour générer les labels d'entraînement et
                           de validation. Shape (N,)
            lr {float} -- Taux d'apprentissage.
            reg {float} -- Terme de régularisation.
            batch_size {int} -- Batch size.
            lr_decay {float} -- Taux de diminution du taux d'apprentissage 
                                (1.0: aucune diminution).
            epochs {int} -- Nombre d'epochs d'entraînement à effectuer.
            experiment_id {str} -- Identificateur de l'expérience.

        Returns:
            list -- Historique de l'accuracy de validation.
        """

        N = len(y)

        shuffle_indices = np.random.permutation(N)
        X_shuffle = X[shuffle_indices]
        y_shuffle = y[shuffle_indices]

        split_index = int(0.8 * N)

        data = {
            'X_train': X_shuffle[:split_index],
            'y_train': y_shuffle[:split_index],
            'X_val': X_shuffle[split_index:],
            'y_val': y_shuffle[split_index:]
        }

        training_config = {
            'lr': lr,
            'reg': reg,
            'batch_size': batch_size,
            'lr_decay': lr_decay,
            'epochs': epochs
        }

        loss_history, train_accuracy_history, val_accuracy_history = self.train_npdl(data,
                                                                                training_config=training_config,
                                                                                verbose=False,
                                                                                experiment_id=experiment_id)

        return loss_history, train_accuracy_history, val_accuracy_history

    def visualize_experiment_results(self, experiment_id='default_exp'):
        """Permet de visualiser les différentes courbes d'entraînement 
           d'une expérience.

        Keyword Arguments:
            experiment_id {str} -- Identificateur de l'expérience (default: {'default_exp'})
        """

        experiment = self.results_cache[experiment_id]
        if experiment is not None:
            loss, train_acc, val_acc = experiment
            visualize_loss(loss, infos=str(self.hyperparam_cache[experiment_id]))
            visualize_accuracy(train_acc, val_acc, infos=str(self.hyperparam_cache[experiment_id]))

    def save_experiment_results(self, experiment_id='default_exp'):
        """Permet de sauvegarder les résultats d'une expérience 
           ainsi que les hyperparmètres utilisés.

        Keyword Arguments:
            experiment_id {str} -- Identificateur de l'expérience (default: {'default_exp'})
        """

        experiment = self.results_cache[experiment_id]
        if experiment is not None:
            loss, train_acc, val_acc = experiment
            visualize_loss(loss, infos=str(self.hyperparam_cache[experiment_id]), save=experiment_id + "_loss")
            visualize_accuracy(train_acc, val_acc,
                               infos=str(self.hyperparam_cache[experiment_id]), save=experiment_id + "_acc")
