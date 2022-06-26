import time

import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import transforms

from RoadSignDataset import RoadSignDataset
from model import CNN

# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Defining hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 30
NUM_CLASSES = 9

# Defining transforms for training data
train_data_transforms = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.15, 0.15)),
        transforms.RandomAffine(0, shear=10),
        transforms.RandomAffine(0, scale=(0.75, 1.6)),
    ]),
    transforms.ToTensor()
])

# Defining transforms for val data
test_data_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Creating custom datasets
train_dataset = RoadSignDataset(pickle_file="preprocessed/train.p", transform=train_data_transforms)
val_dataset = RoadSignDataset(pickle_file="preprocessed/val.p", transform=test_data_transforms)
test_dataset = RoadSignDataset(pickle_file="preprocessed/test.p", transform=test_data_transforms)

# Generating weights for sampler
train_class_count = np.bincount(train_dataset.labels)
train_weights = 1 / np.array([train_class_count[y] for y in train_dataset.labels])

# Creating sampler for balancing the dataset
train_sampler = WeightedRandomSampler(train_weights, NUM_CLASSES * 20000)

# Create data loader for training validation and test
train_loader = data.DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE, sampler=train_sampler)
val_dataset = data.DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
test_dataset = data.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)


# Defining model and training options
model = CNN(NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss().to(device)


# Function to calculate accuracy
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# Function to perform training of the model
def train(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # Train the model
    model.train()

    for (images, labels) in loader:
        # Sending batch images and labels to gpu if cuda is available
        images = images.to(device)
        labels = labels.to(device)

        opt.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()

        acc = calculate_accuracy(output, labels)

        opt.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


# Function to perform evaluation on the trained model
def evaluate(model, loader, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # Set model to evaluate mode
    model.eval()

    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            acc = calculate_accuracy(output, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def save_model(epoch, model):
    model.eval()
    # Input to the model
    example = torch.randn(1, 3, 112, 112)
    traced_script_module = torch.jit.trace(model.cpu(), example)
    torch.jit.save(traced_script_module, f"model{epoch}.pt")

    model.to(device)

# Plotting number of images of each class before Weighted Sampler.

fig, ax = plt.subplots()
ax.bar(range(NUM_CLASSES), np.bincount(train_dataset.labels), 0.5, color='r')
ax.set_xlabel('Signs')
ax.set_ylabel('Count')
ax.set_title('The count of each sign')
plt.show()

# Plotting random transformed image of each class
plt.figure(figsize=(12, 12))
for c in range(NUM_CLASSES):
    j = 0
    for i in range(len(train_dataset)):
        if train_dataset.labels[i] == c:
            j = i
            break

    plt.subplot(7, 7, c + 1)
    plt.suptitle('Random transformed images', fontsize=20)
    plt.axis('off')
    plt.title('class: {}'.format(c))
    plt.imshow(train_dataset.__getitem__(j)[0].permute(1, 2, 0))

plt.show()

# Plotting distribution using Weighted Sampler
balanced_y_train = torch.LongTensor([])

with torch.no_grad():
    for _, y in train_loader:
        balanced_y_train = torch.cat((balanced_y_train, y))

fig, ax = plt.subplots()
ax.bar(range(NUM_CLASSES), np.bincount(balanced_y_train.cpu().numpy()), 0.5, color='r')
ax.set_xlabel('Signs')
ax.set_ylabel('Count')
ax.set_title('The count of each sign')
plt.show()

# Perform training

# List to save training and val loss and accuracies
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

best_val_acc = 0
best_val_loss = float('inf')
rounds_without_improvement = 1

for epoch in range(EPOCHS):
    if rounds_without_improvement > 5 and epoch > 15:
        break

    print("Epoch:", epoch)

    train_start_time = time.monotonic()
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    train_end_time = time.monotonic()

    val_start_time = time.monotonic()
    val_loss, val_acc = evaluate(model, val_dataset, criterion)
    val_end_time = time.monotonic()

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print("Training: Loss = %.6f, Accuracy = %.6f, Time = %.2f seconds" % (
        train_loss, train_acc, train_end_time - train_start_time))
    print("Val: Loss = %.6f, Accuracy = %.6f, Time = %.2f seconds" % (
        val_loss, val_acc, val_end_time - val_start_time))

    # Updating condition variables for early stopping of model training
    if val_acc > best_val_acc or val_loss < best_val_loss:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        rounds_without_improvement = 0

        save_model(epoch, model)
    else:
        rounds_without_improvement += 1

    time.sleep(10)


# Plot for accuracy over the run of all epochs for a better visual understanding
plt.plot(range(1, len(train_acc_list) + 1), train_acc_list)
plt.plot(range(1, len(val_acc_list) + 1), val_acc_list)
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.savefig('acc.png')
plt.show()

# Plot for loss over the run of all epochs for a better visual understanding
plt.plot(range(1, len(train_loss_list) + 1), train_loss_list)
plt.plot(range(1, len(val_loss_list) + 1), val_loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.show()

