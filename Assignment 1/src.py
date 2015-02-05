import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from numpy import pi, exp, sqrt, dot
from numpy import linspace, array, random
from numpy.linalg import cholesky


def plot_gaussian_1d():
    plt.clf()
    mu_sigma = [(-1, 1), (0, 2), (2, 3)]

    for __gaussian in [gaussian(mu, sigma) for mu, sigma in mu_sigma]:
        plt.plot(__gaussian)
    plt.show()


def gaussian(mu, sigma):
    return 1/(sigma * sqrt(2 * pi)) \
       * exp(- (linspace(-10, 10, 1000) - mu)**2 / (2 * sigma**2))


def plot_gaussian_sample_2d():
    plt.clf()

    sigma = array([[0.3, 0.2], [0.2, 0.2]])
    L = cholesky(sigma)
    mean = array([[1], [2]])

    samples = gaussian_sample(100, mean, L)

    x = [sample[0][0] for sample in samples]
    y = [sample[1][0] for sample in samples]

    plt.scatter(x, y)

    true_mean = (mean[0][0], mean[1][0])
    plt.scatter(true_mean[0], true_mean[1], c='r', marker="x", linewidths=5)
    plt.annotate("True mean",
                 xy=true_mean,
                 xytext=(-80, 10),
                 textcoords="offset points", ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=0'))

    actual_mean = sum(x) / len(x), sum(y) / len(y)
    plt.scatter(actual_mean[0], actual_mean[1], c='c', marker="x", linewidths=5)
    plt.annotate("Actual mean",
                 xy=actual_mean,
                 xytext=(0, -80),
                 textcoords="offset points", ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=0'))
    plt.show()


def gaussian_sample(N, mu, L):
    return [mu + dot(L, random.randn(2, 1)) for _ in xrange(N)]

plot_gaussian_1d()
plot_gaussian_sample_2d()