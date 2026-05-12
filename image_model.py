import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# dataset
full_dataset = datasets.ImageFolder("dataset", transform= transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset)-train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# cnn model
class BreastCancerCNN(nn.Module):
    def __init__(self):
        super(BreastCancerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, X):
        X = self.conv_layers(X)
        X = self.fc_layers(X)
        return X
    # model
model = BreastCancerCNN().to(DEVICE)

    # loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

    # tranning
EPOCHS = 5
for epoch in range (EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _,predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100*correct/total
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f"Loss: {running_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}")
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/image_cancer_model.pth")
print("pytorch model trained sucessfull and saved sucessfull")