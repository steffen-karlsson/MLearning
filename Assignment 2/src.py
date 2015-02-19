import matplotlib.pyplot as plt

from numpy import array, vstack, mean, dot, sqrt
from numpy.linalg import pinv


def load_data(data_file, columns):
    data = []
    with open(data_file, mode="r+") as f:
        for line in f.readlines():
            year_data = []
            split_line = line.split(" ")
            for num in columns:
                year_data.append(float(split_line[num - 1]))
            data.append(year_data)
    return data


def design_matrix_linear_regression(data):
    return vstack(tuple([array([1] + row) for row in data]))


def calculate_wml(design_matrix, t):
    return dot(pinv(design_matrix), t)


def plot_linear_regression_2d(wml, expected_t, actual_t):
    y = sum(actual_t, [])
    x = expected_t.flatten()
    plt.scatter(x, y, c='b', marker='o')
    plt.plot(x, eval('{0}+{1}*x'.format(wml[0][0], wml[1][0])), c='r')
    plt.show()


def calculate_expected_target(data):
    target = []
    for inx in xrange(0, len(data)):
        target.append([mean(data[inx])])
    return array(target)


def rms(data, actual_target, wml):
    return sqrt(sum([(actual_target[idx][0] - y(val, wml)) ** 2 for idx, val in enumerate(data)]) / len(data))


def y(x, w):
    return (w[0] + dot(x, w[1:]))[0]


# II.2.1
actual_t = load_data("sunspotsTrainStatML.dt", [6])
for idx, columns in enumerate([[3, 4], [5], range(1, 6)]):
    data = load_data("sunspotsTrainStatML.dt", columns)
    expected_t = calculate_expected_target(data)
    dm = design_matrix_linear_regression(data)
    wml = calculate_wml(dm, array(expected_t))
    print "RMS: " + str(rms(data, actual_t, wml))
    if idx == 1:
        plot_linear_regression_2d(wml, expected_t, actual_t)
    print wml
