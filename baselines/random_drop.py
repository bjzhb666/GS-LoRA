import torch
import random

def random_drop_weights(model):
    # Iterate over each parameter of the model
    for name, param in model.named_parameters():
        random_drop_rate = random.random() * 0.2
        if 'embedding' in name:
            # The parameters of the Embedding layer are directly set to zero randomly
            shape = param.data.shape
            num_zeros = int(random_drop_rate * shape.numel())
            zero_indices = random.sample(range(shape.numel()), num_zeros)
            param.data.view(-1)[zero_indices] = 0
        else:
            # For other parameters, follow the original logic
            shape = param.data.shape
            num_zeros = int(random_drop_rate * shape.numel())
            zero_indices = random.sample(range(shape.numel()), num_zeros)
            param.data.view(-1)[zero_indices] = 0

    return model