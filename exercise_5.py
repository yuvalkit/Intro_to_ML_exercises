import sys
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as f
from torch import optim
from gcommand_dataset import GCommandLoaderTrain, GCommandLoaderTest, get_classes
import math
import copy


class Model(nn.Module):
    def __init__(self, data_height, data_width, labels_count):
        super(Model, self).__init__()
        self.data_height = data_height
        self.data_width = data_width
        self.labels_count = labels_count

        # the parameters for all the cnn layers
        self.kernel_size = 5
        self.stride = 1
        self.padding = 2
        self.pool_kernel_size = 2
        self.pool_stride = 2

        self.num_of_conv_layers = 5

        # cnn layers out channels sizes
        self.conv0_out_chs = 32
        self.conv1_out_chs = 64
        self.conv2_out_chs = 128
        self.conv3_out_chs = 64
        self.conv4_out_chs = 32

        # the fully connected layers sizes
        self.fc0_size = self.calc_fc0_size()  # 480
        self.fc1_size = 256
        self.fc2_size = 128

        # the cnn layers
        self.cnn0_layer = self.get_cnn_layer(1, self.conv0_out_chs)
        self.cnn1_layer = self.get_cnn_layer(self.conv0_out_chs, self.conv1_out_chs)
        self.cnn2_layer = self.get_cnn_layer(self.conv1_out_chs, self.conv2_out_chs)
        self.cnn3_layer = self.get_cnn_layer(self.conv2_out_chs, self.conv3_out_chs)
        self.cnn4_layer = self.get_cnn_layer(self.conv3_out_chs, self.conv4_out_chs)

        # the fully connected layers
        self.fc_layers = self.get_fc_layers()

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.cnn0_layer(x)
        x = self.cnn1_layer(x)
        x = self.cnn2_layer(x)
        x = self.cnn3_layer(x)
        x = self.cnn4_layer(x)
        x = x.view(-1, self.fc0_size)
        x = self.fc_layers(x)
        return f.log_softmax(x, dim=1)

    def get_cnn_layer(self, in_chs, out_chs):
        return nn.Sequential(
            nn.Conv2d(in_chs, out_chs, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_kernel_size, self.pool_stride)
        )

    def get_fc_layers(self):
        return nn.Sequential(
            nn.Linear(self.fc0_size, self.fc1_size),
            nn.ReLU(),
            nn.Linear(self.fc1_size, self.fc2_size),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.fc2_size, self.labels_count)
        )

    def calc_fc0_size(self):
        return self.calc_size(self.data_height) * self.calc_size(self.data_width) * self.conv4_out_chs

    def calc_size(self, size):
        for i in range(self.num_of_conv_layers):
            size = calc_conv_size(size, self.kernel_size, self.padding, self.stride)
            size = calc_pool_size(size, self.pool_kernel_size, self.pool_stride)
        return size


# calculating the size after a cnn layer
def calc_conv_size(size, kernel_size, padding, stride):
        return math.floor(((size - kernel_size + 2 * padding) / stride) + 1)


# calculating the size after a max pool
def calc_pool_size(size, kernel_size, stride):
        return math.floor(((size - kernel_size) / stride) + 1)


# training epochs times on the train loader
# returns a copy of the model that gave the best accuracy
def training_process(model, train_loader, validation_loader, epochs):
    best_accuracy = 0
    best_model = copy.deepcopy(model)
    for e in range(epochs):
        model.train()
        for x, y in train_loader:
            model.optimizer.zero_grad()
            output = model(x)
            loss = f.cross_entropy(output, y)
            loss.backward()
            model.optimizer.step()
        accuracy = get_validation_accuracy(model, validation_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
    return best_model


# returns the accuracy on the given validation loader
def get_validation_accuracy(model, validation_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in validation_loader:
            output = model(x)
            predication = output.max(1, keepdim=True)[1]
            correct += predication.eq(y.view_as(predication)).cpu().sum()
    return (correct / len(validation_loader.dataset)).item()


# get all the single predictions of the model on the test loader
def get_predictions(model, test_loader):
    predictions = []
    with torch.no_grad():
        for x, file_names in test_loader:
            output = model(x)
            predictions_batch = output.max(1, keepdim=True)[1]
            for i, prediction in enumerate(predictions_batch):
                predictions.append((file_names[i], prediction.item()))
    return predictions


# returns a dictionary between the label index and the class name
def get_idx_to_class(dir_path):
    classes = get_classes(dir_path)
    return {i: classes[i] for i in range(len(classes))}


# writes the given predictions to a file
def write_predictions_to_file(predictions, dir_path):
    idx_to_class = get_idx_to_class(dir_path)
    file = open("test_y", "w")
    for file_name, prediction in predictions:
        file.write(f"{file_name},{idx_to_class[prediction]}\n")
    file.close()


# returns a data loader with the gcommand data set
def get_data_loader(gcommand_loader, path, batch_size, shuffle):
    return data.DataLoader(gcommand_loader(path), batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def get_training_loader(path, batch_size):
    return get_data_loader(GCommandLoaderTrain, path, batch_size, True)


def get_test_loader(test_path, batch_size):
    return get_data_loader(GCommandLoaderTest, test_path, batch_size, False)


def main():
    # the const parameters
    labels_count = 30
    data_height = 161
    data_width = 101

    # hyper-parameters
    epochs = 30
    batch_size = 64

    train_path, validation_path, test_path = sys.argv[1], sys.argv[2], sys.argv[3]

    train_loader = get_training_loader(train_path, batch_size)
    validation_loader = get_training_loader(validation_path, batch_size)
    test_loader = get_test_loader(test_path, batch_size)

    model = Model(data_height, data_width, labels_count)

    model = training_process(model, train_loader, validation_loader, epochs)

    predictions = get_predictions(model, test_loader)

    write_predictions_to_file(predictions, train_path)


main()
