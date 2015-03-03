import matplotlib.pyplot as plt
from pylab import legend, rcParams

from numpy import array, vstack, mean, dot, sqrt, where, loadtxt, cov, add, log
from numpy import identity, transpose, linspace, concatenate
from numpy.linalg import pinv, inv

BETA = 1
rcParams['legend.loc'] = 'best'


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


def plot_linear_regression_2d(train_data, train_actual, test_data, test_actual, wml):
    test_data = array(test_data).flatten()
    train_actual = array(train_actual).flatten()
    train_data = array(train_data).flatten()

    plt.scatter(train_data, train_actual, c='b', marker='o')
    plt.scatter(test_data, test_actual, c='g', marker='x')
    plt.plot(test_data, eval('{0}+{1}*test_data'.format(wml[0][0], wml[1][0])), c='r')
    plt.show()


def y(data, w):
    return dot(w.T, array([1] + data))


def rms(data, t, w):
    return sqrt(sum((t[i] - y(d, w))**2 for i, d in enumerate(data)) / len(data))


def MAP(dm, t, alpha, beta=1):
    map_cov = inv(alpha * identity(len(dm[0])) + beta * dot(transpose(dm), dm))
    return beta * dot(dot(map_cov, transpose(dm)), t)


def split_into_species(data):
    sorted_data = array(sorted(data, key=lambda entry: entry[2]))
    species_indicies = sorted_data[:, 2]

    inx1 = where(species_indicies == 1)[0][0]
    inx2 = where(species_indicies == 2)[0][0]

    return sorted_data[:inx1, 0:2], sorted_data[inx1:inx2, 0:2], sorted_data[inx2:, 0:2]


def calculate_cov(species0, species1, species2):
    covs = array([cov(transpose(species0)), cov(transpose(species1)), cov(transpose(species2))])
    return add(*covs) / (200 - len(covs))


def lda(x, species, sigma, combined_sum=200.0):
    mean_species = mean(species, axis=0)
    prior = log(len(species) / combined_sum)
    inv_sigma = inv(sigma)
    return dot(transpose(x), dot(inv_sigma, mean_species)) \
           - 0.5 * dot(dot(transpose(mean_species), inv_sigma), mean_species) + prior


def best_category(x, all_species):
    sigma = calculate_cov(*all_species)
    ldas = [lda(x, species, sigma) for species in all_species]
    return ldas.index(max(ldas))


def plot_train_test_data(train, test, plot_edge=False):
    def __get_species_color(s):
        if s == 0:
            return 'red'
        elif s == 1:
            return 'blue'
        else:
            return 'orange'

    for len, width, species in train:
        plt.scatter(len, width, color=__get_species_color(species))

    if test is not None:
        for len, width, species in test:
            if plot_edge:
                plt.scatter(len, width, color='grey', edgecolors=__get_species_color(species))
            else:
                plt.scatter(len, width, color='grey')
    plt.show()


def lost_function(test, new_test):
    count = 0
    for idx, val in enumerate(test):
        new_val = new_test[idx]
        if new_val[2] != val[2]:
            count += 1
    return count / float(len(test))


def normalize(data):
    lens, heights, species = zip(*data)
    sum_lens = sum(lens)
    sum_heights = sum(heights)

    lens = [val / sum_lens for val in lens]
    heights = [val / sum_heights for val in heights]
    return array([array([lens[i], heights[i], species[i]]) for i in xrange(len(species))])


test = loadtxt("IrisTest2014.dt")
train = loadtxt("IrisTrain2014.dt")

# II.1.1
all_species = split_into_species(train)
new_train = train.copy()
for data in new_train:
    data[2] = best_category(data[0:2], all_species)
plot_train_test_data(train, new_train, plot_edge=True)
print str(lost_function(train, new_train))


new_test = test.copy()
for data in new_test:
    data[2] = best_category(data[0:2], all_species)
plot_train_test_data(train, new_test, plot_edge=True)
print str(lost_function(test, new_test))


# II.1.2
normalized_data = normalize(concatenate((train, test)))
normalized_train = normalized_data[:len(train)]
normalized_test = normalized_data[len(train):]

normalized_all_species = split_into_species(normalized_train)
new_test = normalized_test.copy()

new_train = normalized_train.copy()
for data in new_train:
    data[2] = best_category(data[0:2], normalized_all_species)
plot_train_test_data(normalized_train, new_train, plot_edge=True)
print "Lost: " + str(lost_function(normalized_train, new_train))

for data in new_test:
    data[2] = best_category(data[0:2], normalized_all_species)
plot_train_test_data(normalized_train, new_test, plot_edge=True)
print "Lost: " + str(lost_function(normalized_test, new_test))


# II.2.1
train_actual_t = load_data("sunspotsTrainStatML.dt", [6])
test_actual_t = load_data("sunspotsTestStatML.dt", [6])

for idx, columns in enumerate([[3, 4], [5], range(1, 6)]):
    train_data = load_data("sunspotsTrainStatML.dt", columns)
    dm = design_matrix_linear_regression(train_data)
    wml = calculate_wml(dm, array(train_actual_t))
    test_data = load_data("sunspotsTestStatML.dt", columns)

    print "RMS: " + str(rms(test_data, test_actual_t, wml))
    if idx == 1:
        plot_linear_regression_2d(train_data, train_actual_t, test_data, test_actual_t, wml)
    print wml

xs = linspace(1916, 2011, 2011-1915)
for idx, columns in enumerate([[3, 4], [5], range(1, 6)]):
    train_data = load_data("sunspotsTrainStatML.dt", columns)
    dm = design_matrix_linear_regression(train_data)
    wml = calculate_wml(dm, array(train_actual_t))
    test_data = load_data("sunspotsTestStatML.dt", columns)

    if idx == 0:
        color = 'r'
    elif idx == 1:
        color = 'b'
    else:
        color = 'g'

    plt.plot(xs, array([y(d, wml) for d in test_data]), label="Section {0}".format(idx + 1), c=color)

test_actual_t = array(test_actual_t).flatten()
plt.plot(xs, test_actual_t, c='k', label="True data")
plt.gca().set_xlim([1916, 2011])
legend(framealpha=0.5)
plt.show()

# II.2.2
for idx, columns in enumerate([[3, 4], [5], range(1, 6)]):
    train_data = load_data("sunspotsTrainStatML.dt", columns)
    test_data = load_data("sunspotsTestStatML.dt", columns)
    dm = design_matrix_linear_regression(train_data)

    rmss = []
    alphas = linspace(0, 5000, 2000)
    for alpha in alphas:
        pm = MAP(dm, train_actual_t, alpha)
        rmss.append(rms(test_data, test_actual_t, pm))

    optimal_alpha = alphas[rmss.index(min(rmss))]
    print "Optimal Alpha: " + str(optimal_alpha)
    print "RMS: " + str(min(rmss))
    plt.plot(alphas, rmss, c='b')
    plt.axis([0, 5000, 10, 60])
    plt.show()
