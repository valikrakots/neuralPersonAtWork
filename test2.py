import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F




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






transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])



test_dataset = datasets.ImageFolder('test', transform=transform)
test_set = DataLoader(test_dataset, shuffle=False, batch_size=1)



with torch.no_grad():
    model = torch.load('torch_video_net.pth')
    model.eval()

    total_correct = 0.0

    for inputs, labels in test_set:
        output = model(inputs)
        output_idx = torch.argmax(output, dim=1)
        total_correct += sum(labels == output_idx)

    print("Accuracy: " + str(total_correct) + " out of " + str(len(test_set)))


