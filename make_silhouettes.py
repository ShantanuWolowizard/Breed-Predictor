from torchvision import datasets
train_data = datasets.ImageFolder("dataset/train")
print(train_data.classes)
