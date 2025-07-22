import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shpae [batch_size, 201, 1024]
        x = x.mean(dim=1) # 對時間維度取平均
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x