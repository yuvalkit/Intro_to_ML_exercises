import sys
import numpy as np


# convert the value from nominal to numerical
def convert_nominal(value):
    return 0.0 if value == "W" or value == b"W" else 1.0


# make a x vectors array from a given file
def make_x_array(file_name):
    return np.loadtxt(file_name, delimiter=",", converters={11: convert_nominal})


# make a y array from a given file
def make_y_array(file_name):
    return np.loadtxt(file_name, dtype=int)


# add a column of ones to the given array
def add_bias_column(arr):
    return np.append(arr, np.ones([len(arr), 1]), axis=1)


# combine the two given arrays
def combine_arrays(arr1, arr2):
    return np.array(list(zip(arr1, arr2)), dtype=object)


# calculates the euclidean distance between a and b
def euclidean_distance(a, b):
    return np.linalg.norm(np.subtract(a, b))


# z-score-normalize the given v by the given mean and std
def z_score_normalize(v, mean, std):
    new_v = v - mean
    if std != 0:
        new_v /= std
    return new_v


# z-score-normalize the given array by the given mean vector and std vector
def z_score_normalize_array(arr, mean_v, std_v, feature_count):
    for i in range(feature_count):
        arr[:, i] = z_score_normalize(arr[:, i], mean_v[i], std_v[i])


# z-score-normalize the given train and test
def z_score_normalization(train_x, test_x, feature_count):
    mean_v = np.mean(train_x, axis=0)
    std_v = np.std(train_x, axis=0)
    z_score_normalize_array(train_x, mean_v, std_v, feature_count)
    z_score_normalize_array(test_x, mean_v, std_v, feature_count)


# calculates the accuracy between arr1 and arr2
def get_accuracy(arr1, arr2):
    size = len(arr1)
    if size == 0:
        return 0
    match_count = 0
    for a1, a2 in zip(arr1, arr2):
        if a1 == a2:
            match_count += 1
    return match_count / size


# calculates tau by the known formula
def get_tau(w, x, y, y_hat):
    w_y_x = np.dot(w[y, :], x)
    w_y_hat_x = np.dot(w[y_hat, :], x)
    tau = max(0, 1 - w_y_x + w_y_hat_x)
    denominator = 2 * pow(np.linalg.norm(x), 2)
    if denominator != 0:
        tau /= denominator
    return tau


# the knn algorithm as learned in class
def knn(training_set, test_x, k):
    result = []
    for x in test_x:
        distances = []
        for x_i, y_i in training_set:
            distances.append([euclidean_distance(x, x_i), y_i])
        distances = np.array(distances)
        distances = distances[distances[:, 0].argsort()]
        k_closest = distances[:k]
        k_ys = k_closest[:, 1].astype(np.int)
        y_hat = np.bincount(k_ys).argmax()
        result.append(y_hat)
    return np.array(result)


# the perceptron algorithm as learned in class
# returns the best w and best accuracy on the validation from all the epochs
def perceptron_train(training_set, validation, feature_count, label_count, eta, epochs):
    best_w = np.zeros((label_count, feature_count))
    best_accuracy = 0
    validation_x = validation[:, 0]
    validation_y = validation[:, 1]
    w = np.random.rand(label_count, feature_count)
    for e in range(epochs):
        np.random.shuffle(training_set)
        for x, y in training_set:
            y_hat = np.argmax(np.dot(w, x))
            eta_x = eta * x
            if y_hat != y:
                w[y, :] += eta_x
                w[y_hat, :] -= eta_x
        y_hats = predict_x_with_w(validation_x, w)
        accuracy = get_accuracy(validation_y, y_hats)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_w = w
    return best_w, best_accuracy


# the pa algorithm as learned in class
# returns the learned w and it's accuracy on the validation
def pa_train(training_set, validation, feature_count, label_count):
    validation_x = validation[:, 0]
    validation_y = validation[:, 1]
    w = np.random.rand(label_count, feature_count)
    np.random.shuffle(training_set)
    for x, y in training_set:
        w_without_y = np.delete(w, y, axis=0)
        y_hat = np.argmax(np.dot(w_without_y, x))
        if y_hat >= y:
            y_hat += 1
        tau = get_tau(w, x, y, y_hat)
        tau_x = tau * x
        w[y, :] += tau_x
        w[y_hat, :] -= tau_x
    y_hats = predict_x_with_w(validation_x, w)
    accuracy = get_accuracy(validation_y, y_hats)
    return w, accuracy


# returns the w that gives the best accuracy from all the k-fold cross validation iterations
# train by perceptron if is_perceptron=True, train by pa otherwise
def get_best_w(training_set, feature_count, label_count, is_perceptron, eta, epochs):
    k_fold_k = 10
    best_w = np.zeros((label_count, feature_count))
    best_accuracy = 0
    np.random.shuffle(training_set)
    sub_arrays = np.array(np.array_split(training_set, k_fold_k), dtype=object)
    for i in range(k_fold_k):
        validation = sub_arrays[i].copy()
        train = np.concatenate(np.delete(sub_arrays.copy(), i, axis=0))
        if is_perceptron:
            w, accuracy = perceptron_train(train, validation, feature_count, label_count, eta, epochs)
        else:
            w, accuracy = pa_train(train, validation, feature_count, label_count)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_w = w
    return best_w


# predict all the x's of the given test_x with the given w
def predict_x_with_w(test_x, w):
    predictions = []
    for x in test_x:
        predictions.append(np.argmax(np.dot(w, x)))
    return np.array(predictions)


def main():
    # data parameters
    feature_count = 12
    label_count = 3
    # hyper-parameters
    knn_k = 7
    perceptron_eta = 0.0003
    perceptron_epochs = 80
    # get the data from the given files
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    train_x = make_x_array(train_x)
    train_y = make_y_array(train_y)
    test_x = make_x_array(test_x)
    # normalize the data by z-score-normalization
    z_score_normalization(train_x, test_x, feature_count)
    # add the bias column to the x data
    train_x = add_bias_column(train_x)
    test_x = add_bias_column(test_x)
    feature_count += 1
    # make a training set from the train x and y
    training_set = combine_arrays(train_x, train_y)
    # predict with knn
    knn_predictions = knn(training_set, test_x, knn_k)
    # train and predict with perceptron
    perceptron_w = get_best_w(training_set, feature_count, label_count, True, perceptron_eta, perceptron_epochs)
    perceptron_predictions = predict_x_with_w(test_x, perceptron_w)
    # train and predict with pa
    pa_w = get_best_w(training_set, feature_count, label_count, False, None, None)
    pa_predictions = predict_x_with_w(test_x, pa_w)
    # print all the predictions
    for knn_yhat, perceptron_yhat, pa_yhat in zip(knn_predictions, perceptron_predictions, pa_predictions):
        print(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, pa: {pa_yhat}")


main()
