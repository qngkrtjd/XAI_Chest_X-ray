import torch.nn as nn
import torchvision

class DenseNet121(nn.Module):
    
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121()
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 14),
            nn.Sigmoid()
            #nn.Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        return x