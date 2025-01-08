import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# 1. Define CNN model
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # [batch, 1, 28, 28] -> [batch, 32, 28, 28]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # [batch, 32, 28, 28] -> [batch, 64, 28, 28]
        self.pool = nn.MaxPool2d(2, 2)                           # [batch, 64, 28, 28] -> [batch, 64, 14, 14]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)           # [batch, 64, 14, 14] -> [batch, 128, 12, 12]
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # 2. Prepare MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. Initialize network and optimizer
    model = ComplexCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. Train
    epochs = 10  # Reduced epochs to avoid overfitting
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    scripted_model = torch.jit.script(model)
    scripted_model.save("mnist_complex_cnn.pt")
    print("Model saved to mnist_complex_cnn.pt")

if __name__ == "__main__":
    main()
