"""Random-initialized binary Hey Iris network; no learned external frontend."""
import torch
from torch import nn


class WakeNet(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.config={'width':width}
        self.stem=nn.Sequential(nn.Conv1d(40,width,5,stride=2,padding=2),nn.BatchNorm1d(width),nn.SiLU())
        self.blocks=nn.ModuleList([nn.Sequential(
            nn.Conv1d(width,width,5,padding=2*d,dilation=d,groups=width),nn.BatchNorm1d(width),
            nn.Conv1d(width,2*width,1),nn.SiLU(),nn.Conv1d(2*width,width,1),nn.BatchNorm1d(width))
            for d in (1,2,4,8)])
        self.classifier=nn.Linear(2*width,2)

    def forward(self,features):
        x=self.stem(features)
        for block in self.blocks:x=torch.nn.functional.silu(x+block(x))
        return self.classifier(torch.cat((x.mean(2),x.amax(2)),dim=1))
