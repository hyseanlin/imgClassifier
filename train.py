import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import vgg16, vgg19, resnet50, densenet121
import argparse
from AlexLikeNet import AlexLikeNet
from RestNetLikeNet import ResNet18LikeNet
from CustomDataset import CustomImageDataset
import numpy as np

# Create the parser
parser = argparse.ArgumentParser(description="A simple command-line program")

# Add the arguments
parser.add_argument(
    '--data-dir',
    # required=True,
    help='訓練資料目錄',
    default='data',
)
parser.add_argument(
    '--model-type',
    choices=('VGG16', 'VGG19', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'custom1', 'custom2'),
    default='ResNet50',
    help='選擇模型類別',
)
parser.add_argument(
    '--epochs',
    type=int,
    default=32,
    help='訓練回合數',
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=4,
    help='批次大小',
)
parser.add_argument(
    '--input-width',
    type=int,
    default=256,
    help='模型輸入寬度',
)
parser.add_argument(
    '--input-height',
    type=int,
    default=256,
    help='模型輸入高度',
)
parser.add_argument(
    '--weights-file',
    # required=True,
    help='模型參數檔案',
    default='model.pth',
)
# Parse the arguments
args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
learning_rate = 1e-3

def choose_model(model_type, class_count):
    if model_type == 'VGG16':
        model = vgg16(pretrained=False, num_classes=class_count)
    elif model_type == 'VGG19':
        model = vgg19(pretrained=False, num_classes=class_count)
    elif model_type == 'ResNet50':
        model = resnet50(pretrained=False, num_classes=class_count)
    elif model_type == 'DenseNet121':
        model = densenet121(pretrained=False, num_classes=class_count)
    elif model_type == 'custom1':
        model = AlexLikeNet(class_count)
    else:
        model = ResNet18LikeNet(class_count)
    # Add softmax layer to the end of the model
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    return model

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def main():
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.input_height, args.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

    # Create an instance of the CustomDataset
    train_dataset = CustomImageDataset(annotations_file='train_data/annotations.csv', transform=transform, target_transform=target_transform)
    test_dataset = CustomImageDataset(annotations_file='test_data/annotations.csv', transform=transform, target_transform=target_transform)

    # Create data loaders for training and testing sets using samplers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Print the number of elements in the dataset
    print(f'Total batches for training: {len(train_loader)}')
    print(f'Total batches for testing: {len(test_loader)}')

    # Model selection
    model = choose_model(args.model_type, class_count=train_dataset.classes)
    print(model)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")

    # Save the model
    torch.save(model.state_dict(), args.weights_file)

if __name__ == '__main__':
    main()
