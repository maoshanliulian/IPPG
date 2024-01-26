import numpy as np
import torch
from torch import nn

class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.model=nn.Sequential(
            nn.BatchNorm1d(1),#1 250
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=11,stride=1,padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(5,stride=1,padding=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, stride=1, padding=1),
        )
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True),
        )
        self.GRU = nn.Sequential(
            nn.GRU(input_size=256, hidden_size=64)
        )
        self.fc = nn.Sequential(

            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(16064, 128),
            nn.LeakyReLU(),
            nn.Linear(128,1)
        )


    def forward(self,x):
        x =self.model(x)
        x =x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ =self.GRU(x)
        x = self.fc(x)
        return x