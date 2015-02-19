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


def plot_linear_regression_2d(data, wml, target):
    y = sum(data, [])
    y1 = sum(target, [])
    x = array(range(0, len(y)))
    plt.scatter(x, y, c='b', marker='o')
    plt.scatter(x[16:], y1, c='g', marker='x')
    plt.plot(x, eval('{0}+{1}*x'.format(wml[0][0], wml[1][0])), c='r')
    plt.show()


def calculate_target_values(data):
    target = []
    for inx in xrange(16, len(data)):
        target.append([mean([data[inx - 16][i], data[inx - 8][i], data[inx - 4][i],
                            data[inx - 2][i], data[inx - 1][i]]) for i in xrange(len(data[inx]))])
    return target


def rms(data, target, wml):
    return sqrt(sum([(target[idx] - dot(array([1] + val), wml)) ** 2 for idx, val in enumerate(data)])) / len(data)


# II.2.1
for idx, columns in enumerate([[3, 4], [5], range(1, 6)]):
    data = load_data("sunspotsTrainStatML.dt", columns)
    target = calculate_target_values(data)
    dm = design_matrix_linear_regression(data[16:])
    wml = calculate_wml(dm, array(target))
    print "RMS: " + str(rms(data[16:], target, wml))
    if idx == 1:
        plot_linear_regression_2d(data, wml, target)
    print wml
