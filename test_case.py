import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os

# Define transform to convert images to Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

# Download the MNIST dataset (test set only)
mnist_test = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create output folder for images
output_folder = './mnist_test_images'
os.makedirs(output_folder, exist_ok=True)

# Save test images as PNG files
for idx in range(80):
    image, label = mnist_test[idx]
    image = transforms.ToPILImage()(image)  # Convert tensor to PIL image
    image.save(f"{output_folder}/mnist_digit_{idx}_label_{label}.png")

print(f"Saved {len(mnist_test)} images in '{output_folder}'")
