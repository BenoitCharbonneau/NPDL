# Code adapté de projets académiques de la professeur Fei Fei Li et
# de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Première version rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin.
# Version finale rédigée par Benoit Charbonneau et Pierre-Luc Parent


import pickle as pickle
import numpy as np
import os


def load_CIFAR_batch_file(filename):
    """ charge une batch de cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        data = datadict['data']
        labels = datadict['labels']
        data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        labels = np.array(labels)
        return data, labels


def load_CIFAR10(ROOT):
    """ charge la totalité de cifar """
    all_data = []
    all_labels = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        data, labels = load_CIFAR_batch_file(f)
        all_data.append(data)
        all_labels.append(labels)
    concat_data = np.concatenate(all_data)
    concat_labels = np.concatenate(all_labels)
    del data, labels
    data_test, labels_test = load_CIFAR_batch_file(os.path.join(ROOT, 'test_batch'))
    return concat_data, concat_labels, data_test, labels_test


def load_tiny_imagenet(path, num_classes):
    from PIL import Image
    """
    Charge TinyImageNet. TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 ont la même structure de répertoires, cette fonction peut
    donc être utilisée pour charger n'importe lequel d'entre eux.

    Inputs:
    - path: String du path vers le répertoire à charger.
    - dtype: numpy datatype utilisé pour charger les données.

    Returns: Un tuple de
    - class_names: list, class_names[i] étant une liste de string donnant les
      noms WordNet pour classe i dans le dataset.
    - X_train: (N_tr, 3, 64, 64) array, contient les images d'entraînement
    - y_train: (N_tr,) array, contient les labels d'entraînement
    - X_val: (N_val, 3, 64, 64) array, contient les images de validation
    - y_val: (N_val,) array, contient les labels de validation
    - X_test: (N_test, 3, 64, 64) array, contient le images de test.
    - y_test: (N_test,) array, contient les labels de test; si les labels ne
    sont pas disponibles, y_test = None
    """
    # Charge wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids aux labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Utilise words.txt pour obtenir les noms de chaque classe
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict()
        for line in f:
            items = line.split(',')
            wnid_to_words[items[0]] = [i.strip() for i in items[1:]]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Charge les données d'entraînement.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d' % (i + 1, len(wnids)))

        if i == num_classes:
            break

        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 32, 32), dtype=np.float64)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = Image.open(img_file)
            img.thumbnail((32, 32))
            img_array = np.array(img)
            if img_array.ndim == 2:
                # image grayscale
                img_array.shape = (32, 32, 1)
            X_train_block[j] = img_array.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]
    validation = int(len(y_train) * 0.8)

    return class_names, X_train[:validation], y_train[:validation], X_train[validation:], y_train[validation:]
