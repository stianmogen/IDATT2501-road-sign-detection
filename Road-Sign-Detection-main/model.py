import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.stn = Stn()
        self.logits = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(inplace=True),

                                    nn.Flatten(),
                                    nn.Dropout(0.5),
                                    nn.Linear(256 * 7 * 7, 1000),
                                    nn.ReLU(inplace=True),

                                    nn.Dropout(0.5),
                                    nn.Linear(in_features=1000, out_features=256),
                                    nn.ReLU(inplace=True),

                                    nn.Linear(256, output_dim))

    def forward(self, x):
        x = self.stn(x)
        return self.logits(x)


class Stn(nn.Module):
    def __init__(self):
        super(Stn, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 24 * 24, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1) * xs.size(2) * xs.size(3))
        # calculate the transformation parameters theta
        theta = self.fc_loc(xs)
        # resize theta
        theta = theta.view(-1, 2, 3)
        # grid generator => transformation on parameters theta
        grid = F.affine_grid(theta, x.size())
        # grid sampling => applying spatial transformations
        x = F.grid_sample(x, grid)
        return x