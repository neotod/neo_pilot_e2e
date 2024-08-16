import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_size, in_channels, n_conv_layers):
        super().__init__()
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers

        # Convolutional layers
        conv_layers = []
        self.conv_layers_channels = [in_channels]
        self.conv_layers_channels.extend(
            [16*(2**i) for i in range(n_conv_layers)]
        )
        for i in range(n_conv_layers):
          conv_layers.append(
              nn.Sequential(
                nn.Conv2d(
                    self.conv_layers_channels[i],
                    self.conv_layers_channels[i+1],
                    kernel_size=3, padding=1
                ),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
                nn.Dropout2d(p=0.1)
              )
          )
        self.conv_layers = nn.Sequential(*conv_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(
            self.conv_layers_channels[-1] * (input_size[0] // (2**self.n_conv_layers)) * (input_size[1] // (2**self.n_conv_layers)),
            128
        )
        self.fc1_dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(128, 1)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, self.conv_layers_channels[-1] * (self.input_size[0] // (2**self.n_conv_layers)) * (self.input_size[1] // (2**self.n_conv_layers)))

        x = self.relu(self.fc1(x))
        x = self.fc1_dropout(x)

        x = self.fc2(x)

        return x


class ConvNet2(nn.Module):
    def __init__(self, input_size, in_channels, n_conv_layers):
        super().__init__()
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers

        # Convolutional layers
        conv_layers = []
        self.conv_layers_channels = [in_channels]
        self.conv_layers_channels.extend(
            [16*(2**i) for i in range(n_conv_layers)]
        )
        for i in range(n_conv_layers):
          conv_layers.append(
              nn.Sequential(
                nn.Conv2d(
                    self.conv_layers_channels[i],
                    self.conv_layers_channels[i+1],
                    kernel_size=3, padding=1
                ),
                nn.MaxPool2d(2, 2),
                nn.ReLU()
              )
          )
        self.conv_layers = nn.Sequential(*conv_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(
            self.conv_layers_channels[-1] * (input_size[0] // (2**self.n_conv_layers)) * (input_size[1] // (2**self.n_conv_layers)),
            512
        )
        self.fc2 = nn.Linear(
            512,
            128
        )

        self.fc3 = nn.Linear(128, 1)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, self.conv_layers_channels[-1] * (self.input_size[0] // (2**self.n_conv_layers)) * (self.input_size[1] // (2**self.n_conv_layers)))

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ConvNet54(nn.Module):
    def __init__(self, input_size, in_channels, n_conv_layers):
        super().__init__()
        self.input_size = input_size
        self.n_conv_layers = n_conv_layers

        # Convolutional layers
        conv_layers = []
        self.conv_layers_channels = [in_channels]
        self.conv_layers_channels.extend(
            [16*(2**i) for i in range(n_conv_layers)]
        )
        for i in range(n_conv_layers):
          conv_layers.append(
              nn.Sequential(
                nn.Conv2d(
                    self.conv_layers_channels[i],
                    self.conv_layers_channels[i+1],
                    kernel_size=3, padding=1
                ),
                nn.MaxPool2d(2, 2),
                nn.ReLU()
              )
          )
        self.conv_layers = nn.Sequential(*conv_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(
            self.conv_layers_channels[-1] * (input_size[0] // (2**self.n_conv_layers)) * (input_size[1] // (2**self.n_conv_layers)),
            1024
        )
        self.fc2 = nn.Linear(
            1024,
            512
        )
        self.fc3 = nn.Linear(
            512,
            128
        )
        self.output = nn.Linear(128, 1)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, self.conv_layers_channels[-1] * (self.input_size[0] // (2**self.n_conv_layers)) * (self.input_size[1] // (2**self.n_conv_layers)))

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return self.output(x)
