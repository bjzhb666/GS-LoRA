import torch
import random

def random_drop_weights(model):
    # 遍历模型的每个参数
    for name, param in model.named_parameters():
        random_drop_rate = random.random() * 0.2
        if 'embedding' in name:
            # 对于Embedding层的参数，直接随机置零
            shape = param.data.shape
            num_zeros = int(random_drop_rate * shape.numel())
            zero_indices = random.sample(range(shape.numel()), num_zeros)
            param.data.view(-1)[zero_indices] = 0
        else:
            # 对于其他参数，按照原来的逻辑处理
            shape = param.data.shape
            num_zeros = int(random_drop_rate * shape.numel())
            zero_indices = random.sample(range(shape.numel()), num_zeros)
            param.data.view(-1)[zero_indices] = 0

    return model