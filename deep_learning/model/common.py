from deep_learning.lib.utility import *

def drop_connect(x, probability, training):
    if not training: return x

    batch_size = len(x)
    keep_probability = 1 - probability
    noise = keep_probability
    noise += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    mask = torch.floor(noise)
    x = x / keep_probability * mask

    return x
def upsize_add(x, lateral):
    return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral

def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x
