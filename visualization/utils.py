# Code adapté de projets académiques de la professeur Fei Fei Li et
# de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Première version rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin.
# Version finale rédigée par Benoit Charbonneau et Pierre-Luc Parent

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt


def visualize_as_grid(Xs, ubound=255.0, padding=1):
    """
    Redimensionne un tenseur en 4D pour faciliter la visualisation.

    Inputs:
    - Xs: Numpy array, shape (N, H, W, C)
    - ubound: Les données en sortie vont être entre normalisées entre [0, ubound]
    - padding: Le nombre de pixels entre chaque élément
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


def visualize_loss(loss_history, y_label='Training loss', x_label='Iterations',
                   title='Loss history', infos="", save=""):
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111, autoscale_on=True)
    ax.plot(loss_history, lw=3)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if infos != "":
        ax.text(len(loss_history)/10, np.max(loss_history), infos)
    if save == "":
        plt.show()
    else:
        fig.savefig(save + ".png")


def visualize_accuracy(training_accuracy, validation_accuracy, y_label='Classification accuracy',
                       x_label='Epoch', title='Classification accuracy history', infos="", save=""):
    fig = plt.figure(2, figsize=(8, 5))
    ax = fig.add_subplot(111, autoscale_on=True)
    ax.plot(training_accuracy, label='train', lw=3)
    ax.plot(validation_accuracy, label='val', lw=3)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend()
    if infos != "":
        ax.text(len(training_accuracy)/10, np.max(training_accuracy), infos)
    if save == "":
        plt.show()
    else:
        fig.savefig(save + ".png")
