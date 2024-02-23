import torch.nn as nn
from enum import Enum
import re

import inspect

def delegates(to=None, keep=False):
    # Decorator: replace `**kwargs` in signature with params from `to`
    def _f(f):
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__
        else:          to_f,from_f = to,f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f

    def store_attr(self, nms):
        # Store params named in comma-seperated 'nms'fromcalling context into attrs in 'self'
        mod = inspect.currentframe().f_back.f_locals
        for n in re.split(', *', nms): setattr(self, n,mod[n])
    
    NormType = Enum('NormType', 'Batch Batch')

class ConvLayer(nn.Sequential):
    def __init__():
        return