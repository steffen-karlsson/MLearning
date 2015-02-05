import matplotlib.pyplot as plt

from numpy import pi, exp, sqrt, linspace


def plot_gaussian():
    mu_sigma = [(-1, 1), (0, 2), (2, 3)]

    for __gaussian in [gaussian(mu, sigma) for mu, sigma in mu_sigma]:
        plt.plot(__gaussian)
    plt.show()


def gaussian(mu, sigma):
    return 1/(sigma * sqrt(2 * pi)) \
       * exp(- (linspace(-10, 10, 1000) - mu)**2 / (2 * sigma**2))


plot_gaussian()