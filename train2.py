import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


batch_size = 16


transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

#print(device)

train_dataset = datasets.ImageFolder('train', transform=transform)
train_set = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (3,3))
        self.fc1 = nn.Linear(51984, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()#.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters())

for epoch in range(10):  # loop over the dataset multiple times

    total_loss = 0.0
    for i, (inputs, labels) in enumerate(train_set):
        # zero the parameter gradients
        inputs, labels = inputs, labels
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {total_loss / 20:.3f}')
            total_loss = 0.0

print('Finished Training')

PATH = './torch_video_net.pth'
torch.save(net, PATH)