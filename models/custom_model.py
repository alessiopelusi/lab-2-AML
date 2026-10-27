from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Define layers of the neural network
        self.features = nn.Sequential(
            # input: (3, 64, 64) = 12.288 valori
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # params: (3*3*3 + 1)*64 = 1.792
            nn.ReLU(),        # (64, 64, 64) = 262.144 attivazioni
            nn.MaxPool2d(2),  # (64, 32, 32) = 65.536 attivazioni, 0 params

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # params: (3*3*64 + 1)*128 = 73.856
            nn.ReLU(),        # (128, 32, 32) = 131.072 attivazioni
            nn.MaxPool2d(2),  # (128, 16, 16) = 32.768 attivazioni, 0 params

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# params: (3*3*128 + 1)*256 = 295.168
            nn.ReLU(),        # (256, 16, 16) = 65.536 attivazioni
            nn.MaxPool2d(2),  # (256, 8, 8) = 16.384 attivazioni, 0 params
        )
        self.flatten = nn.Flatten()             # (16.384,), 0 params, cambia solo la forma
        self.ff = nn.Linear(256 * 8 * 8, 200)   # params: 16.384*200 + 200 = 3.277.000
                                                # output: (200,) logit, uno per classe
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)

        return self.ff(x)