import torch
import torch.nn as nn
import random


def random_drop_weights(model, drop_probability):
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        model_without_parallel = model.module
        random_drop_weights(model_without_parallel, drop_probability)
    else:
        for name, module in model.named_children():
            if isinstance(module, nn.Module):
                random_drop_weights(module, drop_probability)
            elif isinstance(module, nn.Parameter):
                if torch.is_tensor(module.data):
                    if random.random() < drop_probability:
                        module.data *= 0
    return model