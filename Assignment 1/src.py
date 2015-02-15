import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from numpy import pi, exp, sqrt, dot, multiply, insert, arange
from numpy import linspace, array, random, newaxis, var
from numpy.linalg import cholesky, eigh, inv

from collections import Counter

from math import radians, cos, sin, hypot

N = 100
MEAN = array([[1], [2]])


def plot_gaussian_1d():
    plt.clf()
    mu_sigma = [(-1, 1), (0, 2), (2, 3)]

    for __gaussian in [gaussian(mu, sigma) for mu, sigma in mu_sigma]:
        plt.plot(linspace(-10, 10, 1000), __gaussian)
    plt.show()


def gaussian(mu, sigma):
    return 1/(sigma * sqrt(2 * pi)) \
       * exp(- (linspace(-10, 10, 1000) - mu)**2 / (2 * sigma**2))


def plot_gaussian_sample_2d(samples, mean):
    plt.clf()

    plot_sample_with_color(samples, 'blue')

    x = [s[0][0] for s in samples]
    y = [s[1][0] for s in samples]

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

    vals, vecs = eigh(ml_covariance(samples, mean))
    scaled_vecs = [mean + sqrt(vals[i]) * vecs[i, newaxis].T
                   for i in xrange(len(vecs))]

    scaled_x = [scaled_vec[0][0] for scaled_vec in scaled_vecs]
    scaled_y = [scaled_vec[1][0] for scaled_vec in scaled_vecs]

    true_x = true_mean[0]
    true_y = true_mean[1]
    plt.arrow(true_x, true_y, scaled_x[0] - true_x, scaled_y[0] - true_y, fc="y", ec="y")
    plt.arrow(true_x, true_y, scaled_x[1] - true_x, scaled_y[1] - true_y, fc="y", ec="y")

    plt.show()


def plot_gaussian_sample_rotated(samples, mean):
    plt.clf()

    ml = ml_covariance(samples, mean)
    for degree, color in [(0, 'blue'), (30, 'red'), (60, 'green'), (90, 'orange')]:
        rotated = rotated_covariance(degree, ml)

        sample = gaussian_sample(N, MEAN, rotated)
        plot_sample_with_color(sample, color)

    plt.show()


def ml_covariance(samples, mean):
    return sum([multiply((sample - mean), (sample - mean).T) for sample in samples]) / len(samples)


def rotated_covariance(degree, ml):
    r_degree = radians(degree)
    R = array([[cos(r_degree), -sin(r_degree)],[sin(r_degree), cos(r_degree)]])
    return (inv(R).dot(ml)).dot(R)


def gaussian_sample(N, mu, sigma):
    L = cholesky(sigma)
    return [mu + dot(L, random.randn(2, 1)) for _ in xrange(N)]


def plot_sample_with_color(sample, c):
    x = [s[0][0] for s in sample]
    y = [s[1][0] for s in sample]
    plt.scatter(x, y, color=c)


sigma = array([[0.3, 0.2], [0.2, 0.2]])
samples = gaussian_sample(N, MEAN, sigma)

plot_gaussian_1d()
# plot_gaussian_sample_2d(samples, MEAN)
# plot_gaussian_sample_rotated(samples, MEAN)


def load_data(data_file):
    data = []
    with open(data_file, mode="r+") as f:
        for line in f.readlines():
            split_line = line.split(" ")
            data.append((float(split_line[0]), float(split_line[1]), int(split_line[2])))
    return data


def plot_train_test_data(train, test):
    for len, width, species in train:
        if species == 0:
            c = 'red'
        elif species == 1:
            c = 'blue'
        else:
            c = 'orange'

        plt.scatter(len, width, color=c)

    if test is not None:
        for len, width, species in test:
            plt.scatter(len, width, color='grey')

    # plt.gca().set_aspect('equal')
    plt.show()


def nearest_neighbour(k, train, test):
    new_test = []
    for test_len, test_width, _ in test:
        distances = sorted([(hypot(test_len - train_len, test_width - train_width), species)
                            for train_len, train_width, species in train])
        most = Counter([distances[i][1] for i in xrange(k)]).most_common()
        new_test.append((test_len, test_width, most[0][0]))
    return new_test


def lost_function(test, new_test):
    count = 0
    for idx, val in enumerate(test):
        new_val = new_test[idx]
        if new_val[2] != val[2]:
            count += 1
    return count / float(len(test))


def cross_validation(train, fold):
    size_train = len(train)
    sizes = array([size_train / fold for _ in xrange(fold)])
    for i in xrange(size_train % fold):
        sizes[i] += 1
    sizes = insert(sizes, 0, 0)

    chunks = [train[sum(sizes[:i]):sum(sizes[:i+1])]
              for i in xrange(1, fold + 1)]

    optimal_ks = []
    for k in xrange(1, 26, 2):
        loses = 0
        for idx, chunk in enumerate(chunks):
            rest = sum(chunks[:idx] + chunks[idx + 1:], [])
            new_test = nearest_neighbour(k, rest, chunk)
            loses += lost_function(chunk, new_test)
        optimal_ks.append((k, loses / len(chunks)))

    x, y = zip(*optimal_ks)
    plt.plot(x, y)
    plt.show()


def normalize(data):
    lens, heights, species = zip(*data)
    sum_lens = sum(lens)
    sum_heights = sum(heights)

    print (sum_lens / len(lens), var(lens)), \
           (sum_heights / len(heights), var(heights))

    lens = [val / sum_lens for val in lens]
    heights = [val / sum_heights for val in heights]

    print (sum(lens) / len(lens), var(lens)), \
           (sum(heights) / len(heights), var(heights))

    return zip(lens, heights, species)


test = load_data("IrisTest2014.dt")
train = load_data("IrisTrain2014.dt")

# new_test = nearest_neighbour(5, train, test)
# plot_train_test_data(train, test)
# plot_train_test_data(new_test, None)
# print str(1 - lost_function(test, new_test))
#cross_validation(train, fold=5)

#updated_data = normalize(train)
#cross_validation(updated_data, fold=5)
