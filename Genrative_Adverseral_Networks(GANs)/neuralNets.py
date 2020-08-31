import torch
import torch.nn as nn

class Descriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Descriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer1 = nn.Sequential(
                        nn.Conv2d(self.in_channels, 64*2, 
                            kernel_size=4, stride=2, padding=1
                            ),
                        nn.LeakyReLU(0.2)
                    )

        self.layer2 = nn.Sequential(
                        nn.Conv2d(64*2, 128*2,
                            kernel_size=4, stride=2, padding=1,
                            bias=False
                            ),
                        nn.BatchNorm2d(128*2),
                        nn.LeakyReLU(0.2)
                    )

        self.layer3 = nn.Sequential(
                        nn.Conv2d(128*2, 256*2,
                            kernel_size=4, stride=2, padding=1,
                            bias=False),
                        nn.BatchNorm2d(256*2),
                        nn.LeakyReLU(0.2)
                    )

        self.layer4 = nn.Sequential(
                        nn.Conv2d(256*2, 512*2,
                            kernel_size=4, stride=2, padding=1,
                            bias=False),
                        nn.BatchNorm2d(512*2),
                        nn.LeakyReLU(0.2),
                    )

        self.layer5 = nn.Sequential(
                        nn.Conv2d(512*2, 1,
                            kernel_size=4, stride=2, padding=0
                            ),
                        nn.Sigmoid()
                    )

        self.dropout_layer = nn.Sequential(
                                nn.Dropout(0.45)
                            )
        
        
    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        #output = self.dropout_layer(output)
        output = self.layer5(output)

        return output
            


class Generator(nn.Module):
    def __init__(self, noise_channels, out_channels):
        super(Generator, self).__init__()
        self.noise_channel = noise_channels
        self.out_channels = out_channels

        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(noise_channels, 512*2,
                            kernel_size=4, stride=1, padding=0,
                            bias=False
                            ),
                        nn.BatchNorm2d(512*2),
                        nn.ReLU()
                    )

        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(512*2, 256*2,
                            kernel_size=4, stride=2, padding=1,
                            bias=False
                            ),
                        nn.BatchNorm2d(256*2),
                        nn.ReLU()
                    )

        self.layer3 = nn.Sequential(
                        nn.ConvTranspose2d(256*2, 128*2,
                            kernel_size=4, stride=2, padding=1,
                            bias=False
                            ),
                        nn.BatchNorm2d(128*2),
                        nn.ReLU()
                    )

        self.layer4 = nn.Sequential(
                        nn.ConvTranspose2d(128*2, 64*2,
                            kernel_size=4, stride=2, padding=1,
                            bias=False
                            ),
                        nn.BatchNorm2d(64*2),
                        nn.ReLU()
                    )

        self.layer5 = nn.Sequential(
                        nn.ConvTranspose2d(64*2, self.out_channels,
                            kernel_size=4, stride=2, padding=1
                            ),
                        nn.Tanh()
                    )

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)

        return output