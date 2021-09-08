import sys
import torch
from torch.utils import data
from torch import optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets
from torchvision import transforms


# the exercise const parameters
PIXEL_RANGE = 255
IMAGE_SIZE = 784
LABELS_COUNT = 10
EPOCHS = 10


# data set for train sets
class TrainDataSet(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# data set for test sets
class TestDataSet(data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


# model A
class ModelA(nn.Module):
    def __init__(self, lr):
        super(ModelA, self).__init__()
        self.hid_layer1_size = 100
        self.hid_layer2_size = 50
        self.lr = lr
        self.fc0 = nn.Linear(IMAGE_SIZE, self.hid_layer1_size)
        self.fc1 = nn.Linear(self.hid_layer1_size, self.hid_layer2_size)
        self.fc2 = nn.Linear(self.hid_layer2_size, LABELS_COUNT)
        self.optimizer = self.get_optimizer()

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        return f.log_softmax(self.fc2(x), -1)

    def get_optimizer(self):
        return optim.SGD(self.parameters(), lr=self.lr)


# model B
class ModelB(ModelA):
    def __init__(self, lr, beta0, beta1):
        self.betas = (beta0, beta1)
        super().__init__(lr)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)


# model C
class ModelC(ModelA):
    def __init__(self, lr, p0, p1):
        super().__init__(lr)
        self.dropout0 = nn.Dropout(p0)
        self.dropout1 = nn.Dropout(p1)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = self.dropout0(f.relu(self.fc0(x)))
        x = self.dropout1(f.relu(self.fc1(x)))
        return f.log_softmax(self.fc2(x), -1)


# model D
class ModelD(ModelA):
    def __init__(self, lr):
        super().__init__(lr)
        self.bn0 = nn.BatchNorm1d(self.hid_layer1_size)
        self.bn1 = nn.BatchNorm1d(self.hid_layer2_size)
        self.bn2 = nn.BatchNorm1d(LABELS_COUNT)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = f.relu(self.bn0(self.fc0(x)))
        x = f.relu(self.bn1(self.fc1(x)))
        return f.log_softmax(self.bn2(self.fc2(x)), -1)


# model E
class ModelE(nn.Module):
    def __init__(self, lr, beta0, beta1):
        super(ModelE, self).__init__()
        self.lr = lr
        self.betas = (beta0, beta1)
        self.hid_layer1_size = 128
        self.hid_layer2_size = 64
        self.hid_layer3_size = 10
        self.hid_layer4_size = 10
        self.hid_layer5_size = 10
        self.fc0 = nn.Linear(IMAGE_SIZE, self.hid_layer1_size)
        self.fc1 = nn.Linear(self.hid_layer1_size, self.hid_layer2_size)
        self.fc2 = nn.Linear(self.hid_layer2_size, self.hid_layer3_size)
        self.fc3 = nn.Linear(self.hid_layer3_size, self.hid_layer4_size)
        self.fc4 = nn.Linear(self.hid_layer4_size, self.hid_layer5_size)
        self.fc5 = nn.Linear(self.hid_layer5_size, LABELS_COUNT)
        self.fcs = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4]
        self.optimizer = self.get_optimizer()
        self.activation_function = f.relu

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        for fc in self.fcs:
            x = self.activation_function(fc(x))
        return f.log_softmax(self.fc5(x), -1)

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)


# model F
class ModelF(ModelE):
    def __init__(self, lr, beta0, beta1):
        super().__init__(lr, beta0, beta1)
        self.activation_function = torch.sigmoid


# doing one pass of training on the train loader
def train_one_epoch(model, train_loader):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        model.optimizer.zero_grad()
        output = model(x)
        loss = f.nll_loss(output, y)
        loss.backward()
        model.optimizer.step()


# training epochs times on the train loader
def training_process(model, train_loader):
    for e in range(EPOCHS):
        train_one_epoch(model, train_loader)


# return the loss and accuracy of the model on the given loader
def get_loader_results(model, loader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            total_loss += f.nll_loss(output, y, size_average=False).item()
            predication = output.max(1, keepdim=True)[1]
            correct += predication.eq(y.view_as(predication)).cpu().sum()
    size = len(loader.dataset)
    loss = total_loss / size
    accuracy = (correct / size).item()
    return loss, accuracy


# normalizes the array by dividing its values by 255
def normalize(arr):
    arr /= PIXEL_RANGE
    return arr


# merge the 2 given arrays
def merge_examples_and_predictions(examples, predictions):
    return np.array(list(zip(examples, predictions)), dtype=object)


# convert the numpy array to torch tensor
def convert_np_to_tensors(np_arr):
    return torch.from_numpy(np_arr)


# return the shuffled array
def shuffle(arr):
    np.random.shuffle(arr)
    return arr


# split the training set to train and validation by the split_k ratio
def split_training_set(training_set, split_k):
    parts = np.array(np.array_split(training_set.copy(), split_k), dtype=object)
    validation = parts[0].copy()
    train = np.concatenate(np.delete(parts.copy(), 0, axis=0))
    return train, validation


# convert the given wrapped numpy array to torch tensor
def get_tensor_from_np(np_arr):
    np_arr = np_arr.reshape(len(np_arr), 1)
    np_arr = np.array([np_arr[i][0] for i in range(len(np_arr))])
    return convert_np_to_tensors(np_arr)


# return a data loader from the given numpy set (x and y)
def get_data_loader(np_set, batch_size, to_shuffle):
    tensor_x = get_tensor_from_np(np_set[:, 0]).clone().detach().float()
    tensor_y = get_tensor_from_np(np_set[:, 1]).clone().detach().long()
    data_set = TrainDataSet(tensor_x, tensor_y)
    return data.DataLoader(data_set, batch_size=batch_size, shuffle=to_shuffle)


# train the model on the train and return loss and accuracy on the validation
def train_and_get_results(model, train_loader, validation_loader):
    training_process(model, train_loader)
    loss, accuracy = get_loader_results(model, validation_loader)
    return loss, accuracy


# return validation accuracy and train accuracy per epoch in 'validation_accuracies' and 'train_accuracies' arrays
# return validation loss and train loss per epoch in 'validation_losses' and 'train_losses' arrays
# return the accuracy on the FashionMNIST test set after all epochs in 'test_set_accuracy'
def get_all_results(training_set, batch_size, model, split_k):
    validation_accuracies = []
    train_accuracies = []
    validation_losses = []
    train_losses = []
    training_set = shuffle(training_set)
    train_set, validation_set = split_training_set(training_set, split_k)
    train_loader = get_data_loader(train_set, batch_size, True)
    validation_loader = get_data_loader(validation_set, batch_size, False)
    for e in range(EPOCHS):
        train_one_epoch(model, train_loader)
        validation_loss, validation_accuracy = get_loader_results(model, validation_loader)
        train_loss, train_accuracy = get_loader_results(model, train_loader)
        validation_accuracies.append(validation_accuracy)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        train_losses.append(train_loss)
    test_set_accuracy = get_test_set_accuracy(batch_size, model)
    return validation_accuracies, train_accuracies, validation_losses, train_losses, test_set_accuracy


# return the model accuracy on the FashionMNIST test set
def get_test_set_accuracy(batch_size, model):
    transform = transforms.Compose([transforms.ToTensor()])
    fashion_mnist_data_set = datasets.FashionMNIST('./data', train=False, transform=transform)
    test_loader = data.DataLoader(fashion_mnist_data_set, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            predication = output.max(1, keepdim=True)[1]
            correct += predication.eq(y.view_as(predication)).cpu().sum()
    accuracy = (correct / len(test_loader.dataset)).item()
    return accuracy


# train the model and then writing the test_x predictions to a file
def train_and_predict_test_x(training_set, test_x, model, batch_size, split_k):
    training_set = shuffle(training_set)
    train_set, validation_set = split_training_set(training_set, split_k)
    train_loader = get_data_loader(train_set, batch_size, True)
    training_process(model, train_loader)
    test_tensor = convert_np_to_tensors(test_x).clone().detach().float()
    test_loader = data.DataLoader(TestDataSet(test_tensor), batch_size=batch_size, shuffle=False)
    predictions = get_predictions(model, test_loader)
    write_predictions_to_file(predictions)


# get all the single predictions of the model on the test loader
def get_predictions(model, test_loader):
    predictions = []
    with torch.no_grad():
        for x in test_loader:
            output = model(x)
            predictions_batch = output.max(1, keepdim=True)[1]
            for prediction in predictions_batch:
                predictions.append(prediction.item())
    return predictions


# write the given predictions to a file
def write_predictions_to_file(predictions):
    file = open("test_y", "w")
    for prediction in predictions:
        file.write(f"{prediction}\n")
    file.close()


def main():
    # all models hyper-parameters
    # model A
    model_a_batch_size = 4
    model_a_lr = 0.012
    # model B
    model_b_batch_size = 32
    model_b_lr = 0.001
    model_b_beta0 = 0.9
    model_b_beta1 = 0.99
    # model C
    model_c_batch_size = 8
    model_c_lr = 0.045
    model_c_p0 = 0.2
    model_c_p1 = 0.2
    # model D
    model_d_batch_size = 32
    model_d_lr = 0.032
    # model E
    model_e_batch_size = 8
    model_e_lr = 0.002
    model_e_beta0 = 0.85
    model_e_beta1 = 0.99
    # model F
    model_f_batch_size = 8
    model_f_lr = 0.003
    model_f_beta0 = 0.9
    model_f_beta1 = 0.98
    # load and normalize the data
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    train_x = np.loadtxt(train_x)
    test_x = np.loadtxt(test_x)
    train_y = np.loadtxt(train_y, dtype=int)
    train_x = normalize(train_x)
    test_x = normalize(test_x)
    # merge train_x and train_y
    training_set = merge_examples_and_predictions(train_x, train_y)
    # model B is the best model we chose to use
    model_b = ModelB(model_b_lr, model_b_beta0, model_b_beta1)
    # split_k is 5 in order to get 80:20 train and validation ratio
    split_k = 5
    # training the model and then writing the test_x predictions to a file
    train_and_predict_test_x(training_set, test_x, model_b, model_b_batch_size, split_k)


if __name__ == '__main__':
    main()
 
