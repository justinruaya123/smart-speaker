"""One direct acoustic encoder; flat control or modular action/slot heads."""
import torch
from torch import nn
from torch.nn import functional as F
from scripts.models.vcm import VCM, valid_frames
from scripts.runtime.demo_slots_contract import COMMANDS, ACTIONS


class DemoSlotVCM(nn.Module):
    def __init__(self, mode='heads'):
        super().__init__()
        if mode not in ('flat','heads'):
            raise ValueError('Unknown shared demo mode')
        self.mode = mode
        self.network = VCM(classes=len(COMMANDS) if mode == 'flat' else len(ACTIONS), width=128,
            kernels=(7,7,7,7),dilations=(1,2,4,8),dropout=.1,pooling='mean_max')
        if mode == 'heads':
            self.percent_head = nn.Linear(self.network.head.in_features,4)
            self.alarm_head = nn.Linear(self.network.head.in_features,4)
        self.output_size = len(COMMANDS) if mode == 'flat' else len(ACTIONS)+8
        self.config = {**self.network.config,'mode':mode,'input_temporal_stride':2,
            'output_size':self.output_size,'unknown_class':0,'parameter_limit_exclusive':220000,
            'head_sizes':[len(COMMANDS)] if mode == 'flat' else [len(ACTIONS),4,4]}
        if sum(p.numel() for p in self.parameters()) >= 220000:
            raise ValueError('Shared VCM exceeds the complete parameter cap')

    def forward(self, x, lengths):
        valid_frames(x,lengths)
        embedding = self.network.embedding(x[...,::2],(lengths+1)//2)
        action = self.network.head(embedding)
        if self.mode == 'flat':
            return action
        return torch.cat((action,self.percent_head(embedding),self.alarm_head(embedding)),dim=1)


def supervised_loss(scores, labels, mode, auxiliary_weight=.5):
    if mode == 'flat':
        return F.cross_entropy(scores,labels['class_index'])
    if mode != 'heads':
        raise ValueError('Unknown loss mode')
    # Missing/not-applicable is explicitly supervised as zero; real percent zero is class1.
    return (F.cross_entropy(scores[:,:len(ACTIONS)],labels['action'])
        + auxiliary_weight * F.cross_entropy(scores[:,len(ACTIONS):len(ACTIONS)+4],labels['percent'])
        + auxiliary_weight * F.cross_entropy(scores[:,-4:],labels['alarm']))
