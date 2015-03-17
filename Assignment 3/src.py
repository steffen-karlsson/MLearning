import matplotlib.pyplot as plt

from numpy import loadtxt, array, sqrt, sin, average, zeros
from numpy import subtract, dot, add, mean, var, arange
from numpy.random import rand, seed, sample

from copy import copy

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

pTestData = loadtxt("data/parkinsonsTestStatML.dt", usecols=tuple(range(0, 21)))
pTestTarget = loadtxt("data/parkinsonsTestStatML.dt", usecols=(22, ))
pTrainData = loadtxt("data/parkinsonsTrainStatML.dt", usecols=tuple(range(0, 21)))
pTrainTarget = loadtxt("data/parkinsonsTrainStatML.dt", usecols=(22, ))

sTrainData = loadtxt("data/sincTrain25.dt", usecols=(0, ))
sTrainTarget = loadtxt("data/sincTrain25.dt", usecols=(1, ))
sValidateData = loadtxt("data/sincValidate10.dt", usecols=(0, ))
sValidateTarget = loadtxt("data/sincValidate10.dt", usecols=(1, ))

seed(10)

num_hidden = 2
weights_md = sample([num_hidden, 2])
weights_km = sample([1, num_hidden + 1])
trainInterval = arange(-10, 10, 0.05, dtype='float64')
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]


def input_unit(a):
    return a


def output_unit(hidden_data, weight, bias):
    return dot(weight, hidden_data + [bias])


def alt_sigmoid(a):
    return a / float(1 + abs(a))


def alt_sigmoid_prime(a):
    return 1 / float((1 + abs(a)))**2


def hidden_unit(data, bias, weight):
    return alt_sigmoid(weight[0] * data + weight[1] * bias)


def bias_unit():
    return 1


def nn(data, weights_md, weights_km, num_hidden, num_input=1, num_output=1):
    out_entry = []
    for entry in data:
        hidden_data = []
        for h_unit in xrange(num_hidden):
            hidden_data.append(hidden_unit(entry, bias_unit(), weights_md[h_unit]))
        out_entry.append(output_unit(hidden_data, weights_km, bias_unit()))
    return out_entry


def plot(data, target, estimated):
    plt.scatter(data, target, c='b')
    if estimated is not None:
        plt.scatter(data, estimated, c='r')
    plt.show()


def MSE(data, t):
    return sum((t[i] - data[i])**2 for i, d in enumerate(data)) / len(data)


def weighed_sum(data, weight):
    return sum([datai * weight[i] for i, datai in enumerate(data)])


def delta_k(estimated, target):
    return estimated - target


def delta_j(aj, weight, delta_k):
    return alt_sigmoid_prime(aj) * sum([weight * data_k for k, data_k in enumerate(delta_k)])


def back_prop(train, target, steps, learning_rate, weights_md, weights_km):
    errors = []

    for _ in xrange(steps):
        estimated = nn(train, weights_md, weights_km, num_hidden)
        estimated_interval = nn(trainInterval, weights_md, weights_km, num_hidden)

        dhiddens = []
        douts = []
        for i, data in enumerate(train):
            dK = delta_k(estimated[i], target[i])

            dJs = []
            list_zj = []
            for j in xrange(num_hidden):
                aj = weights_md[j][0] * data + weights_md[j][1] * bias_unit()
                list_zj.append(alt_sigmoid(aj))
                dJs.append(delta_j(aj, weights_km[0][j], [dK]))

            dhiddens.append(dot(array(dJs), array([[data, bias_unit()]])))
            list_zj.append(array([alt_sigmoid(1)]))
            douts.append(dK * array(list_zj))

        avg_dhidden = average(dhiddens, axis=0)
        avg_dout = [list(average(douts, axis=0).flatten())]
        weights_md = subtract(weights_md, (learning_rate * avg_dhidden))
        weights_km = subtract(weights_km, avg_dout)

        errors.append(MSE(estimated, target))
    return errors, estimated_interval, avg_dhidden, avg_dout


def standardization(data, features=None):
    updated_data = []
    new_features = []
    data = zip(*data)

    # print "Mean (Data): " + str(mean(data, axis=1))
    # print "Variance (Data): " + str(var(data, axis=1))

    for idx, d in enumerate(data):
        if features:
            d_var = features[idx][0]
            d_mean = features[idx][1]
        else:
            d_var = sqrt(var(d))
            d_mean = mean(d)
            new_features.append((d_var, d_mean))
        updated_data.append(array([(entry - d_mean) / d_var for entry in d]))

    updated_data = array(updated_data)
    # print "Mean (Norm data): " + str(mean(updated_data, axis=1))
    # print "Variance (Norm data): " + str(var(updated_data, axis=1))

    return updated_data.T, new_features


def estimate_svm_params(data, target, c_range, y_range):
    svr = SVC(kernel='rbf')
    params = {'C': c_range, 'gamma': y_range}
    clf = GridSearchCV(svr, params, cv=5)
    clf.fit(data, target)
    best_params = clf.best_params_
    return best_params['C'], best_params['gamma']


def svm(data, target, c, y):
    svr = SVC(kernel='rbf', C=c, gamma=y)
    svr.fit(data, target)
    return svr


def svm_classify(svr, data):
    return svr.predict(data)


def gradient_verify(data, data_target, weight_km, weight_md, e):
    error_matrix_md = zeros(weights_md.shape)
    error_matrix_km = zeros(weights_km.shape)

    errors, _, _, _ = back_prop(data, data_target, 1, -1, weights_md, weights_km)

    for i in xrange(weights_md.shape[1]):
        for j in xrange(len(weights_md)):
            cpy_weight_md = copy(weights_md)
            cpy_weight_md[j][i] += e
            e_errors, _, _, _ = back_prop(data, data_target, 1, -1, cpy_weight_md, weights_km)
            error_matrix_md[i][j] = (e_errors[0][0] - errors[0][0]) / e

    for i in xrange(len(weights_km)):
        cpy_weight_km = copy(weights_km)
        cpy_weight_km[i] += e
        e_errors, _, _, _ = back_prop(data, data_target, 1, -1, weight_md, cpy_weight_km)
        error_matrix_km[i] = (e_errors[0][0] - errors[0][0]) / e
    return error_matrix_md, error_matrix_km


def lost_function(test, new_test):
    count = 0
    for idx, val in enumerate(test):
        if new_test[idx] != val:
            count += 1
    return count / float(len(test))


def one_loss_svm_classification(train_data, train_target,
                                test_data, test_target):
    c_range = arange(9, 11.01, 0.01)
    y_range = [0.001, 0.01, 0.1, 1, 10, 100]
    c_best, y_best = estimate_svm_params(train_data, train_target,
                                         c_range, y_range)
    print "Best Hyperparameters (C, y): (%.3f , %.3f)" % (c_best, y_best)
    _svm = svm(train_data, train_target, c_best, y_best)

    _classification = svm_classify(_svm, train_data)
    print "Loss function for training data: " + str(lost_function(train_target, _classification))

    _classification = svm_classify(_svm, test_data)
    print "Loss function for test data: " + str(lost_function(test_target, _classification))

## III.1

error_matrix_md, error_matrix_km = gradient_verify(sTrainData[:10], sTrainTarget[:10], weights_km, weights_md, 0.0000000000001)
_, _, avg_dhidden, avg_dout = back_prop(sTrainData[:10], sTrainTarget[:10], 1, -1, weights_md, weights_km)

print subtract(error_matrix_md, avg_dhidden)
print subtract(error_matrix_km, avg_dout)

errors_list = []
data_list = []

for learning_rate in learning_rates:
    errors, data, _, _ = back_prop(sTrainData, sTrainTarget,
                                   10000, learning_rate, weights_md, weights_km)
    errors_list.append(errors)
    data_list.append(data)

plt.gca().set_yscale('log')

for index, err in enumerate(errors_list):
    plt.plot(range(len(err)), err, label="Learning rate " + str(learning_rates[index]))
plt.legend()
plt.show()

for index, da in enumerate(data_list):
    plt.plot(trainInterval, da, label="Learning rate " + str(learning_rates[index]))
plt.plot(trainInterval, eval('sin(trainInterval)/trainInterval'.format(trainInterval)), label="sin(x)/x")
plt.scatter(sTrainData, sTrainTarget)
plt.legend()
plt.show()


## III.2.1
print "Data:"
norm_pTrainData, features = standardization(pTrainData)
one_loss_svm_classification(pTrainData, pTrainTarget,
                            pTestData, pTestTarget)

print "\nNormalized data:"
norm_pTestData, _ = standardization(pTestData, features=features)
one_loss_svm_classification(norm_pTrainData, pTrainTarget,
                            norm_pTestData, pTestTarget)
