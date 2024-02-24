from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        image = self.to_pil(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    annotations_file = 'train_data/annotations.csv'
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create an instance of the CustomDataset
    dataset = CustomImageDataset(annotations_file=annotations_file, transform=transform)

    # Print the number of elements in the dataset
    print(f'Total images: {len(dataset)}')

    # Define the batch size
    batch_size = 32

    # Create a DataLoader instance
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the DataLoader
    for batch in dataloader:
        images, labels = batch
        print(f'Batch images shape: {images.shape}, Batch labels shape: {labels.shape}')

        # Access and print the element in the batch
        image = images[0]
        label = labels[0]
        print(f'In current batch, first image shape: {image.shape}, first image label: {label}')

        # Convert the tensor image to numpy for displaying
        image = image.numpy().transpose((1, 2, 0))
        # Display the image
        plt.imshow(image)
        plt.show()

        break  # We break after the first batch for this example
