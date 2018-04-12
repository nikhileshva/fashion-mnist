import torch
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torch.cuda as cuda

CUDA = cuda.is_available()

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=int, default='0.001', help='Learning Rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch Size')
    return parser.parse_args()

args = get_args()

normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train = FashionMNIST(root='./data', train=True, transform=transform, download=True)
test = FashionMNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=args.batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=args.batch_size,
                                          shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.dropout = nn.Dropout(p=0.5)

        # dense
        self.fc1 = nn.Linear(32*7*7, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)

        return out


model = CNN()

if CUDA:
    model.cuda()

# Loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# train

iter = 0
for epoch in range(args.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        iter += 1
        if (i+1) % 100 == 0:
            print 'Epoch: {}. Iteration: {}. Loss: {}'.format(epoch+1, iter, loss.data[0])


# test model
model.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

accuracy = (100 * correct / total)
print('Test Accuracy of the model on the 10000 test images: {}'.format(accuracy))