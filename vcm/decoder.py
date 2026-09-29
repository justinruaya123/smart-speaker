"""Exact whole-sequence CTC likelihood over finite command pronunciations."""
import numpy as np


class CTCRanker:
    def __init__(self, sequences):
        if not sequences or any(not s or any(type(v) is not int or not 1 <= v < 40 for v in s) for s in sequences):
            raise ValueError('Expected nonempty phone sequences excluding blank')
        width = 2*max(map(len,sequences))+1
        self.labels = np.zeros((len(sequences),width),dtype='int64')
        self.ends = np.array([2*len(s) for s in sequences])
        self.valid = np.arange(width)[None,:] <= self.ends[:,None]
        for k,s in enumerate(sequences): self.labels[k,1:2*len(s):2]=s
        self.skip = np.zeros_like(self.valid)
        self.skip[:,2:] = (self.labels[:,2:] != 0) & (self.labels[:,2:] != self.labels[:,:-2])
        self.reset()

    def reset(self):
        self.alpha = np.full(self.labels.shape, -np.inf)
        self.alpha[:,0]=0
        self.frames=0

    def feed(self, logits):
        x=np.asarray(logits,dtype='float64')
        if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError('Invalid logits')
        x=x-np.max(x,axis=1,keepdims=True)
        logp=x-np.log(np.sum(np.exp(x),axis=1,keepdims=True))
        for frame in logp:
            one=np.full_like(self.alpha,-np.inf); one[:,1:]=self.alpha[:,:-1]
            two=np.full_like(self.alpha,-np.inf); two[:,2:]=self.alpha[:,:-2]
            self.alpha=np.logaddexp(np.logaddexp(self.alpha,one),np.where(self.skip,two,-np.inf))+frame[self.labels]
            self.alpha[~self.valid]=-np.inf
            self.frames+=1
        rows=np.arange(len(self.ends))
        return np.logaddexp(self.alpha[rows,self.ends],self.alpha[rows,self.ends-1])/max(1,self.frames)

    def score(self, logits):
        self.reset(); return self.feed(logits)
