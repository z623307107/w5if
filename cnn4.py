import io
from math import ceil
from statistics import mean
import random
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

TRAIN = 300
EPOCH = 10
BATCHSIZE = 1
LEARNING_RATE = 0.001


device = torch.device('cpu')
print('using:', device)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(BATCHSIZE, 4, 3, 3)
        self.conv2 = nn.Conv1d(4, 8, 4, 1)
        self.fc = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        return self.fc(x.view(-1))

def dataread(data_filename):

    labels = []
    targets = []


    with open(data_filename, 'r') as file:
        for line in file:
            line = [int(i) if i.isdecimal() else int(i, 16) for i in line.split()]
            labels.append(tuple(line[:12]))
            targets.append(sum(line[12:]))


    c = list(zip(labels, targets))
    random.shuffle(c)
    labels[:], targets[:] = zip(*c)

    print('traindata read success')

    return labels[:TRAIN], targets[:TRAIN], labels[TRAIN:], targets[TRAIN:]

def test(model, test_data, test_targets):

    correct = 0

    with torch.no_grad():

        for data, target in zip(test_data, test_targets):
            inputs = torch.tensor(data, dtype=torch.float).to(device).unsqueeze(0)
            targets = torch.tensor(target, dtype=torch.float).to(device).unsqueeze(0)


            output = model(inputs)


            correct += (int(output.item()//9) == target//9)


    return correct / len(test_data) * 100


def train(train_data, train_targets, test_data, test_targets):

    model = CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(EPOCH):
        running_loss = 0.0
        for i in range(TRAIN):
            inputs = torch.tensor(train_data[i], dtype=torch.float).unsqueeze(0).to(device)
            targets = torch.tensor(train_targets[i], dtype=torch.float).unsqueeze(0).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                losses.append(running_loss / 10)
                train_accuracy.append(test(model, train_data, train_targets))
                test_accuracy.append(test(model, test_data, test_targets))
                running_loss = 0.0

    return model, losses, train_accuracy, test_accuracy

if __name__ == '__main__':

    path = 'C:/Users/dawood/Desktop/CNN4/'

    train_data, train_targets, test_data, test_targets = dataread(path + 'data.txt')

    # 批量训练
    for model_name in ['net' + str(i+1).zfill(3) for i in range(20)]:

        print('model:', model_name)


        model, losses, train_accuracy, test_accuracy = train(train_data, train_targets, test_data, test_targets)


        torch.save(model, path + 'net/' + model_name)


        final_accuracy = round(test(model, test_data, test_targets), 2)
        print(f'Accuracy: { final_accuracy }%')


        plt.plot(torch.tensor(losses).detach().numpy(), label='loss')
        plt.plot(torch.tensor(train_accuracy).detach().numpy(), label='train_accuracy')
        plt.plot(torch.tensor(test_accuracy).detach().numpy(), label='test_accuracy')
        plt.legend()
        plt.xlabel('Training Loss (avg of 10 datas)')
        plt.ylabel('Value')
        plt.title('Final Accuracy: %.2f %%' % final_accuracy)
        plt.savefig(path + 'img/' + model_name + '_loss.png')
        plt.close()
        print('save img ok')

    print('done!')
