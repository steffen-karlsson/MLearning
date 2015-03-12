import matplotlib.pyplot as plt

from numpy import loadtxt, array, append, sqrt, inf, arange, sin, round, average
from numpy.random import rand, seed

pTest = loadtxt("data/parkinsonsTestStatML.dt")
pTrain = loadtxt("data/parkinsonsTrainStatML.dt")
sTrainData = loadtxt("data/sincTrain25.dt", usecols=(0, ))
sTrainTarget = loadtxt("data/sincTrain25.dt", usecols=(1, ))
sValidateData = loadtxt("data/sincValidate10.dt", usecols=(0, ))
sValidateTarget = loadtxt("data/sincValidate10.dt", usecols=(1, ))


def input_unit(a):
    return a


def output_unit(hidden_data, weight, bias):
    return weighed_sum([bias] + hidden_data, weight)


def alt_sigmoid(a):
    return a / (1 + abs(a))


def alt_sigmoid_prime(a):
    return 1 / ((1 + abs(a))**2)


def hidden_unit(data, bias, weight):
    return alt_sigmoid(weight[0] * bias + weight[1] * data)


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


def delta_out(target, estimat):
    return estimat - target


def delta_hidden(zj, delta_k, weight):
    return (1 - zj**2) * sum([weight * data_k for k, data_k in enumerate(delta_k)])


def weighed_sum(data, weight):
    return sum([datai * weight[i] for i, datai in enumerate(data)])


def back_prop(steps, learning_rate, weights_md, weights_km):
    errors = []
    prev_error = inf
    current_error = inf
    validation_errors = []
    interval_errors = []

    for _ in xrange(steps):
        delta = prev_error - current_error
        if 0.00001 > delta >= 0:
            break

        estimated = nn(sTrainData, weights_md, weights_km, num_hidden)
        estimated_validation = nn(sValidateData, weights_md, weights_km, num_hidden)
        estimated_interval = nn(trainInterval, weights_md, weights_km, num_hidden)

        dhiddens = []
        douts = []
        for i, data in enumerate(sTrainData):
            delta_k = delta_out(data, estimated[i])

            delta_js = array([])
            list_zj = []
            for j in xrange(num_hidden + 1):
                zj = alt_sigmoid_prime(weights_md[j][0] * bias_unit() + weights_md[j][1] * data)
                list_zj.append(array([zj]))
                delta_js = append(delta_js, delta_hidden(zj, [delta_k], weights_km[j]))

            dhiddens.append((delta_js * array([[bias_unit()], [data]])).T)
            douts.append(delta_k * array(list_zj))

        prev_error = current_error
        # print "PREVIOUS: " + str(prev_error)
        current_error = round(MSE(estimated, sTrainTarget), 3)
        # print "CURRENT: " + str(current_error) + "\n\n"

        weights_md = weights_md - (learning_rate * average(dhiddens, axis=0))
        weights_km = weights_km - (learning_rate * average(douts, axis=0))

        validation_errors.append(MSE(estimated_validation, sValidateTarget))
        interval_errors.append(MSE(estimated_interval,
                                   eval('sin(trainInterval)/trainInterval'.format(trainInterval))))
        errors.append(current_error)
    return errors, validation_errors, interval_errors, estimated_interval


num_hidden = 20

weights_md = rand(num_hidden + 1, 2) - 0.5
weights_km = rand(num_hidden + 1, 1) - 0.5

trainInterval = arange(-10, 10, 0.05, dtype='float64')
errors, validation_errors, interval_errors, data = back_prop(100, 0.001, weights_md, weights_km)
# data = nn(trainInterval, wMD, wKM, num_hidden)

print MSE(data, eval('sin(trainInterval)/trainInterval'.format(trainInterval)))

plt.plot(range(len(interval_errors)), interval_errors)
plt.plot(range(len(errors)), errors)
plt.plot(range(len(validation_errors)), validation_errors)
plt.show()

plt.plot(trainInterval, data)
plt.plot(trainInterval, eval('sin(trainInterval)/trainInterval'.format(trainInterval)))
plt.scatter(sTrainData, sTrainTarget)
plt.show()

# plt.gca().set_yscale('log')
