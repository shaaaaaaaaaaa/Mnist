import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from logisticRegressionModel import LogisticRegressionModel


# # Download Mnist Dataset
transform_funcs = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.1307, ),(0.3081))])

train_dataset = datasets.MNIST(root = './data', train = True, transform =
    transform_funcs,download = False)
test_dataset = datasets.MNIST(root = './data',train = False,transform = transform_funcs)

print("train set size is {}".format(len(train_dataset)))
print("test set size is {}".format(len(test_dataset)))

# Load Data Set
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True,drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False,drop_last=False)

# instantiate the model
n_inputs = 28*28
n_outputs = 10
device = torch.device('cpu')
model = LogisticRegressionModel(n_inputs,n_outputs).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
model_path = "D:/mnist/records"
num_epochs = 100

train_Acc=np.array([])
train_Loss=np.array([])

test_Acc=np.array([])
test_Loss=np.array([])

# train & test
for epoch in range(num_epochs):
    # 训练模型
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    test_loss = 0
    correct_test = 0
    total_test = 0

    for i,(images, labels) in enumerate(train_loader):
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(images.view(-1,28*28))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # record information
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()
        # print(i, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #       % (train_loss / (i + 1), 100. * correct_train / total_train, correct_train, total_train))

    train_Acc = np.append(train_Acc, 100. * correct_train / total_train)
    train_Loss = np.append(train_Loss, train_loss / (i + 1))

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    np.savetxt(model_path + '/result_of_train_acc.txt', train_Acc, fmt='%f')
    np.savetxt(model_path + '/result_of_train_loss.txt', train_Loss, fmt='%f')

    model.eval()
    with torch.no_grad():
        for i,(images, labels) in enumerate(test_loader):
            images,labels = images.to(device),labels.to(device)
            outputs = model(images.view(-1,28*28))
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
            # print(i, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #       % (test_loss / (i + 1), 100. * correct_test / total_test, correct_test, total_test))

        test_Acc = np.append(test_Acc, 100. * correct_test / total_test)
        test_Loss = np.append(test_Loss, test_loss / (i + 1))

        if not os.path.exists(model_path):
            os.mkdir(model_path)
        np.savetxt(model_path + '/result_of_test_acc.txt', test_Acc, fmt='%f')
        np.savetxt(model_path + '/result_of_test_loss.txt', test_Loss, fmt='%f')


torch.save(model.state_dict(), model_path + '/model.pth')
