import matplotlib.pyplot as plt
import seaborn as sns
import torch.onnx
from sklearn.metrics import confusion_matrix
from torch.utils import data
from torchvision import transforms

from RoadSignDataset import RoadSignDataset

torch.cuda.empty_cache()

# Using cuda to utilize GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print(device)

# Load trained model
model = torch.jit.load("models/model9classes.pt").to(device)

num_classes = 9

stn_data_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.RandomAffine(0, shear=5),
    transforms.RandomAffine(0, scale=(0.5, 1.8)),
    transforms.ToTensor()
])

# Defining transforms for val data
test_data_transforms = transforms.Compose([
    transforms.ToTensor()
])

BATCH_SIZE = 128

# Retrieving datasets (validation for displaying STN)
val_dataset = RoadSignDataset(pickle_file="preprocessed/val.p", transform=stn_data_transforms)
test_dataset = RoadSignDataset(pickle_file="preprocessed/test.p", transform=test_data_transforms)
val_dataset = data.DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
test_dataset = data.DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# Function to perform evaluation on the trained model
def evaluate(model, loader):
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            acc = calculate_accuracy(output, labels)

            epoch_acc += acc.item()

    return epoch_acc / len(loader)


test_acc = evaluate(model, test_dataset)
print("Accuracy = %.4f" % test_acc)
print("")


def visualize_stn(model, loader):
    plt_images = []
    model.eval()

    with torch.no_grad():
        # Finding one image of each class
        for (images, labels) in loader:
            for i in range(len(labels)):
                if labels[i] == len(plt_images):
                    plt_images.append(images[i])

        # Plotting grid with images before stn
        fig1 = plt.figure(figsize=(8, 8))
        fig1.suptitle('Images without stn', fontsize=20)
        for i in range(len(plt_images)):
            plt.subplot(7, 7, i + 1)
            plt.axis('off')
            plt.title('class: {}'.format(i))
            plt.imshow(plt_images[i].permute(1, 2, 0).contiguous())
        # Plotting grid with images after stn
        fig2 = plt.figure(figsize=(8, 8))
        fig2.suptitle('Images with stn', fontsize=20)
        for i in range(len(plt_images)):
            plt.subplot(7, 7, i + 1)
            plt.axis('off')
            plt.title('class: {}'.format(i))
            plt.imshow(model.stn(plt_images[i][None, :].to(device))[0].cpu().permute(1, 2, 0).contiguous())
    plt.show()


visualize_stn(model, val_dataset)


def print_confusion_matrix(model, loader):
    y_pred = []
    y_actual = []

    model.eval()

    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)

            output = model(images)

            for p in output.argmax(1):
                y_pred.append(p.cpu().item())
            for a in labels:
                y_actual.append(a)

    cm = confusion_matrix(y_actual, y_pred)

    plt.figure(figsize=(10, 7))

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    plt.savefig(f"confusionmatrix")
    plt.show()


print_confusion_matrix(model, test_dataset)
