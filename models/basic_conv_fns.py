import torch.nn as nn
from torch import cat
import math
from fastai.layers import *
import torch
import torch.nn.functional as F
from typing import Optional, Collection
from fastcore.basics import listify

def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant,irrelevant)
    apr, api = module.ap.attrib(relevant,irrelevant)
    return cat([mpr, apr],1), cat([mpi, api], 1)
    
def attrib_adaptiveconcatpool(self, relevant, irrelevant):
    return cd_adaptiveconcatpool(relevant, irrelevant, self)

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

class AdaptiveConcatPool1d(nn.Module):
    # layer concatting 'AdaptiveAvgPool1d' and 'AdaptiveMaxPool1d'
    def __init__(self, sz:Optional[int]=None):
        # Output will be 2*sz or 2 if sz=None
        super().__init__    ()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    
    def forward(self, x):
        return cat([self.mp(x), self.ap(x)], 1)
    
    def attrib(self, relevant, irrelevant):
        return attrib_adaptiveconcatpool(self, relevant, irrelevant)

def create_head1d(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps=0.5, bn_final:bool=False, bn:bool=True, act='relu', concat_pooling=True):
    # Model head that takes 'nf' features, running through 'lin_ftrs' and about 'nc' classes; added bn and activation
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc]
    ps = listify(ps)
    
    if len(ps)==1:
        ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
        actns = [nn.ReLU(inplace=True) if act=='relu' else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
        layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()] # add adaptive
        
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, bn, p, actn)
    
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    
    return nn.Sequential(*layers)

class ClassificationModel(object):
    def __init__(self,):
        pass
    
    def fit(self, X_train, y_train, X_val, y_val):
        pass
    
    def predict(self, X, full_sequence=True):
        pass

class SqueezeExcite1d(nn.Module):
    '''squeeze excite block as used for example in LSTM FCN'''
    def __init__(self,channels,reduction=16):
        super().__init__()
        channels_reduced = channels//reduction
        self.w1 = torch.nn.Parameter(torch.randn(channels_reduced,channels).unsqueeze(0))
        self.w2 = torch.nn.Parameter(torch.randn(channels, channels_reduced).unsqueeze(0))

    def forward(self, x):
        #input is bs,ch,seq
        z=torch.mean(x,dim=2,keepdim=True)#bs,ch
        intermed = F.relu(torch.matmul(self.w1,z))#(1,ch_red,ch * bs,ch,1) = (bs, ch_red, 1)
        s=F.sigmoid(torch.matmul(self.w2,intermed))#(1,ch,ch_red * bs, ch_red, 1=bs, ch, 1
        return s*x #bs,ch,seq * bs, ch,1 = bs,ch,seq

def weight_init(m):
    '''call weight initialization for model n via n.appy(weight_init)'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    if isinstance(m,SqueezeExcite1d):
        stdv1=math.sqrt(2./m.w1.size[0])
        nn.init.normal_(m.w1,0.,stdv1)
        stdv2=math.sqrt(1./m.w2.size[1])
        nn.init.normal_(m.w2,0.,stdv2)