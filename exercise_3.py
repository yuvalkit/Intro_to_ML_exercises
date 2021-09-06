import sys
import numpy as np


# does a division, if b is 0, return a
def safe_division(a, b):
    return a / b if b != 0 else a


# return sigmoid on x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# return softmax on x
def softmax(x):
    exp_x = np.exp(x)
    return safe_division(exp_x, exp_x.sum())


# shuffle the given array
def shuffle(arr):
    np.random.shuffle(arr)


# return a (a,b) shape array with random values between 0 to 1
def rand(a, b):
    return np.random.rand(a, b)


# convert the given array to numpy array of type object
def np_arr(arr):
    return np.array(arr, dtype=object)


# add a column of ones to the given array
def add_bias_column(arr):
    return np.append(arr, np.ones([len(arr), 1]), axis=1)


# combine the two given arrays
def combine_arrays(arr1, arr2):
    return np_arr(list(zip(arr1, arr2)))


# calculate the accuracy between arr1 and arr2
def get_accuracy(arr1, arr2):
    size = len(arr1)
    if size == 0:
        return 0
    match_count = 0
    for a1, a2 in zip(arr1, arr2):
        if a1 == a2:
            match_count += 1
    return match_count / size


# initialize random w1 and w2 matrices and normalizes them
def init_w1_and_w2(feature_count, label_count, hid_layer_dim):
    w1 = rand(hid_layer_dim, feature_count) / feature_count
    w2 = rand(label_count, hid_layer_dim) / hid_layer_dim
    return w1, w2


# return a one hot encoded vector of the given y
def get_one_hot(y, label_count):
    one_hot_y = np.zeros(label_count)
    one_hot_y[y] = 1
    return one_hot_y.reshape((label_count, 1))


# the forward propagation as learned in class
def forward_propagation(x, w1, w2):
    x = x.reshape((len(x), 1))
    z1 = np.dot(w1, x)
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1)
    h2 = softmax(z2)
    return h1, h2


# the back propagation as learned in class
# dz2 was calculated for the multi-class case
def back_propagation(x, y, w2, h1, h2, label_count):
    one_hot_y = get_one_hot(y, label_count)
    dz2 = h2 - one_hot_y         # dL/dz2
    dw2 = np.dot(dz2, h1.T)      # dL/dz2 * dz2/dw2
    dh1 = np.dot(w2.T, dz2)      # dL/dz2 * dz2/dh1
    dz1 = dh1 * (h1 * (1 - h1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dw1 = np.dot(dz1, x.T)       # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    return dw1, dw2


# predict all the x's of the given x_arr with the given w1 and w2
def predict_labels(x_arr, w1, w2):
    predictions = []
    for x in x_arr:
        h1, h2 = forward_propagation(x, w1, w2)
        predictions.append(np.argmax(h2))
    return np.array(predictions)


# update w1 and w2 by the update rule
def update_w(w1, w2, dw1, dw2, eta):
    w1 -= eta * dw1
    w2 -= eta * dw2
    return w1, w2


# the main training process of the neural network
def training_process(training_set, feature_count, label_count, hid_layer_dim, epochs, eta):
    shuffle(training_set)
    # train size will be 50000, validation size will be 5000
    sub_arrays = np_arr(np.array_split(training_set, 11))
    # get a validation set to check accuracy on
    validation = sub_arrays[0].copy()
    validation_x = validation[:, 0]
    validation_y = validation[:, 1]
    # all the training set without the validation
    train = np.concatenate(np.delete(sub_arrays.copy(), 0, axis=0))
    w1, w2 = init_w1_and_w2(feature_count, label_count, hid_layer_dim)
    # best_w1 and best_w2 will be the w1 and w2 that gave the best accuracy from all the epochs
    best_w1, best_w2 = init_w1_and_w2(feature_count, label_count, hid_layer_dim)
    best_accuracy = 0
    for e in range(epochs):
        shuffle(train)
        for x, y in train:
            # train and update w1 and w2 from the x and y example
            x = x.reshape((len(x), 1))
            h1, h2 = forward_propagation(x, w1, w2)
            dw1, dw2 = back_propagation(x, y, w2, h1, h2, label_count)
            w1, w2 = update_w(w1, w2, dw1, dw2, eta)
        # calculate the current accuracy on the validation
        y_hats = predict_labels(validation_x, w1, w2)
        accuracy = get_accuracy(validation_y, y_hats)
        if accuracy > best_accuracy:
            # update the bests
            best_accuracy = accuracy
            best_w1 = w1.copy()
            best_w2 = w2.copy()
    return best_w1, best_w2


# write all the test_x predictions to a 'test_y' file
def write_y_hats_to_file(y_hats):
    f = open("test_y", "w")
    for y_hat in y_hats:
        f.write(f"{y_hat}\n")
    f.close()


def main():
    # exercise parameters
    feature_count = 785
    label_count = 10
    pixel_range = 255
    # hyper-parameters
    hid_layer_dim = 100
    epochs = 50
    eta = 0.01
    # files arguments
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    # load all the files data
    train_x = np.loadtxt(train_x)
    train_y = np.loadtxt(train_y, dtype=int)
    test_x = np.loadtxt(test_x)
    # normalize the data
    train_x /= pixel_range
    test_x /= pixel_range
    # add bias to the data
    train_x = add_bias_column(train_x)
    test_x = add_bias_column(test_x)
    # make the training set
    training_set = combine_arrays(train_x, train_y)
    # train the neural network
    w1, w2 = training_process(training_set, feature_count, label_count, hid_layer_dim, epochs, eta)
    # predict the test_x labels with the trained w1 and w2
    y_hats = predict_labels(test_x, w1, w2)
    # write the predictions to a file
    write_y_hats_to_file(y_hats)


if __name__ == '__main__':
    main()
