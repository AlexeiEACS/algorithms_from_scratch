"""
Implementation of a Perceptron
Guided from the Book: Machine Learning with PyTorch and Scikit-Learn

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # Se usa w_ ya que por convención los atributos creados fuera del método __init__ llevan de sufijo "_"
        # Inicializamos w_ con una muestra de una normal con ds 0.01
        self.w_ = rgen.normal(loc=0.0,
                              scale=0.01,
                              size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        # Se regresa self porque se están haciendo las modificaciones sobres los atributos
        # También se usa para poder anidar procesos
        return self

    def net_input(self, X):
        """
        Activation Function
        Is a linear combination of wights taht are connectionf the input layer to the output layer.
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return the class label after unit step
        """
        return np.where(self.net_input(X) > 0.0, 1, 0)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')


if __name__ == "__main__":
    """
    Las pruebas de nuestro perceptron se harán sobre el Iris DataSet
    Para poder pasar de una clasificación binaria a una multi-label usaremos la técnica one-versus-rest (OvA)
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print(f'From the url: {url}')

    df = pd.read_csv(url,
                     header=None,
                     encoding='utf8')
    print('El dataframe a entrenar:')
    print(df.head())

    # Vamos a reemplazar los valores de setosa y versicolor por un encoder de 0 o 1
    # Trabajaremos con los primero 100 registros
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    # Ahora extraemos nuestros features que serán sepal lenght y petal lenght
    X = df.iloc[0:100, [0, 2]].values
    print('Data visualization')
    plt.scatter(x=df.iloc[0:100, 0], y=df.iloc[0:100, 2],
                color=np.where(y == 0, 'r', 'blue'), marker='o')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.title("Scatter Plot: Setosa (Red) vs. Versicolor (Blue)")
    plt.show()

    "Ahora si vamos a entrenar a nuestro Perceptron"
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1),
             ppn.errors_, marker='d')
    plt.xlabel('Epoch')
    plt.ylabel('Number of updates')
    plt.show()
    # I increase the resolution to see the boundary
    plot_decision_regions(X, y, classifier=ppn, resolution=0.1)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
