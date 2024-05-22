import mindspore as ms

from mindspore import nn,ops

class FedAvgCNN(nn.Cell):
    def __init__(self,in_features=1, num_classes=10, dim=1024):
        super().__init__()

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      has_bias=True,
                      pad_mode='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2)
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      has_bias=True,
                      pad_mode='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fcs = nn.SequentialCell(
            nn.Dense(dim,512),
            nn.ReLU(),
            nn.Dense(512,num_classes)
        )
    
    def construct(self,x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = ops.flatten(x,start_dim=1)

        x = self.fcs(x)

        return x