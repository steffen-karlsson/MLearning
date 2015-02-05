import matplotlib.pyplot as plt

from numpy import pi, exp, sqrt, linspace


def plot_gaussian():
    def __gaussian(_mu, _sigma):
        return 1/(_sigma * sqrt(2 * pi)) \
               * exp(- (linspace(-10, 10, 1000) - _mu)**2 / (2 * _sigma**2))

    mu_sigma = [(-1, 1), (0, 2), (2, 3)]

    for gaussian in [__gaussian(mu, sigma) for mu, sigma in mu_sigma]:
        plt.plot(gaussian)
    plt.show()


plot_gaussian()