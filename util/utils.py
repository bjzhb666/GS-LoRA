import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from .verification import evaluate

from datetime import datetime
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import numpy as np
from PIL import Image
import mxnet as mx
import io
import os, pickle, sklearn
import time
from IPython import embed
import wandb
import math
import torch.nn as nn
import random
from collections import defaultdict
import torch

# from torch.utils.data import Subset, Dataset
from image_iter import CustomSubset
import loralib as lora


def get_time():
    return (str(datetime.now())[:-10]).replace(" ", "-").replace(":", "-")


def load_bin(path, image_size=[112, 112]):
    bins, issame_list = pickle.load(open(path, "rb"), encoding="bytes")
    data_list = []
    for flip in [0, 1]:
        data = torch.zeros((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = mx.nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = torch.tensor(img.asnumpy())
        if i % 1000 == 0:
            print("loading bin", i)
    print(data_list[0].shape)
    return data_list, issame_list


def get_val_pair(path, name):
    ver_path = os.path.join(path, name + ".bin")
    print("ver_path", ver_path)
    assert os.path.exists(ver_path)
    data_set, issame = load_bin(ver_path)
    print("ver", name)
    return data_set, issame


def get_val_data(data_path, targets):
    assert len(targets) > 0
    vers = []
    for t in targets:
        data_set, issame = get_val_pair(data_path, t)
        vers.append([t, data_set, issame])
    return vers


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if "model" in str(layer.__class__):
            continue
        if "container" in str(layer.__class__):
            continue
        else:
            if "batchnorm" in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find("bn") >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    return paras_only_bn, paras_wo_bn


def separate_mobilefacenet_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if "mobilefacenet" in str(layer.__class__) or "container" in str(
            layer.__class__
        ):
            continue
        if "batchnorm" in str(layer.__class__):
            paras_only_bn.extend([*layer.parameters()])
        else:
            paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    plt.close()

    return buf


def test_forward(device, backbone, data_set):
    backbone = backbone.to(device)
    backbone.eval()  # switch to evaluation mode
    # embed()
    # last_time1 = time.time()
    forward_time = 0
    carray = data_set[0]
    # print("carray:",carray.shape)
    idx = 0
    with torch.no_grad():
        while idx < 2000:
            batch = carray[idx : idx + 1]
            batch_device = batch.to(device)
            last_time = time.time()
            backbone(batch_device)
            forward_time += time.time() - last_time
            # if idx % 1000 ==0:
            #    print(idx, forward_time)
            idx += 1
    print("forward_time", 2000, forward_time, 2000 / forward_time)
    return forward_time


def perform_val(
    multi_gpu,
    device,
    embedding_size,
    batch_size,
    backbone,
    data_set,
    issame,
    nrof_folds=10,
):
    """
    Perform face verification on LFW ect.
    """
    if multi_gpu:
        backbone = backbone.module  # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval()  # switch to evaluation mode

    embeddings_list = []
    for carray in data_set:
        # import pdb; pdb.set_trace()
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])  # embedding_size = 512
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx : idx + batch_size]
                # last_time = time.time()
                embeddings[idx : idx + batch_size] = backbone(batch.to(device)).cpu()
                # batch_time = time.time() - last_time
                # print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray):
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print("embeddings shape", embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return (
        accuracy.mean(),
        accuracy.std(),
        _xnorm,
        best_thresholds.mean(),
        roc_curve_tensor,
    )


def perform_val_deit(
    multi_gpu,
    device,
    embedding_size,
    batch_size,
    backbone,
    dis_token,
    data_set,
    issame,
    nrof_folds=10,
):
    if multi_gpu:
        backbone = backbone.module  # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval()  # switch to evaluation mode

    embeddings_list = []
    for carray in data_set:
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx : idx + batch_size]
                # last_time = time.time()
                # embed()
                fea, token = backbone(batch.to(device), dis_token.to(device))
                embeddings[idx : idx + batch_size] = fea.cpu()
                # batch_time = time.time() - last_time
                # print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray):
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print("embeddings shape", embeddings.shape)  # (12000, 512)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return (
        accuracy.mean(),
        accuracy.std(),
        _xnorm,
        best_thresholds.mean(),
        roc_curve_tensor,
    )


def buffer_val(db_name, acc, std, xnorm, best_threshold, roc_curve_tensor, batch):
    # writer.add_scalar('Accuracy/{}_Accuracy'.format(db_name), acc, batch)
    # writer.add_scalar('Std/{}_Std'.format(db_name), std, batch)
    # writer.add_scalar('XNorm/{}_XNorm'.format(db_name), xnorm, batch)
    # writer.add_scalar('Threshold/{}_Best_Threshold'.format(db_name), best_threshold, batch)
    # writer.add_image('ROC/{}_ROC_Curve'.format(db_name), roc_curve_tensor, batch)
    wandb.log(
        {
            "{}_Accuracy".format(db_name): acc,
            "{}_Std".format(db_name): std,
            "{}_XNorm".format(db_name): xnorm,
            "{}_Best_Threshold".format(db_name): best_threshold,
            #    "{}_ROC_Curve".format(db_name): roc_curve_tensor
        },
        step=batch,
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


'''
def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
'''


def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # embed()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0]


def split_dataset(
    dataset,
    class_order_list,
    split1_start,
    split1_end,
    split2_start,
    split2_end,
    transform=ToTensor(),
):
    # get the number of classes
    num_classes = len(dataset.classes)

    # cut interval 1's class index range
    split1_class_indices = class_order_list[
        split1_start:split1_end
    ]  # Does not include split1_end

    # create a dataset for interval 1
    split1_samples = [
        (sample, label)
        for sample, label in dataset.samples
        if label in split1_class_indices
    ]
    split1_dataset = ImageFolder(root=dataset.root, transform=transform)
    split1_dataset.samples = split1_samples
    split1_dataset.targets = [label for _, label in split1_samples]
    split1_dataset.classes = [dataset.classes[idx] for idx in split1_class_indices]
    split1_dataset.class_to_idx = {
        class_name: i for i, class_name in enumerate(split1_dataset.classes)
    }

    # import pdb; pdb.set_trace()
    # cut interval 2's class index range
    split2_class_indices = class_order_list[split2_start:split2_end]  # [90:100]

    # create a dataset for interval 2
    split2_samples = [
        (sample, label)
        for sample, label in dataset.samples
        if label in split2_class_indices
    ]
    split2_dataset = ImageFolder(root=dataset.root, transform=transform)
    split2_dataset.samples = split2_samples
    split2_dataset.targets = [label for _, label in split2_samples]
    split2_dataset.classes = [dataset.classes[idx] for idx in split2_class_indices]
    split2_dataset.class_to_idx = {
        class_name: i for i, class_name in enumerate(split2_dataset.classes)
    }

    return split1_dataset, split2_dataset


def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def reinitialize_lora_parameters(model):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" in name:
                if isinstance(param, nn.Parameter):
                    if "lora_A" in name:
                        nn.init.kaiming_uniform_(param, a=math.sqrt(50))
                    elif "lora_B" in name:
                        nn.init.zeros_(param)
                else:
                    raise ValueError(
                        f"Parameter {name} is not an instance of nn.Parameter."
                    )


def get_unique_classes(subset, original_dataset):
    """
    Get all unique category names and total number of categories in the subset.

        :param subset: torch.utils.data.Subset object
        :param original_dataset: original complete data set
        :return: (list of category names, total number of categories)
    """
    unique_classes = subset.classes
    total_classes = len(unique_classes)
    return unique_classes, total_classes


def create_few_shot_dataset(dataset, n_shot, seed=None):
    """
    Creates a few-shot version of the dataset containing a specified number of samples per class.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset, which must contain the `targets` attribute.
        n_shot (int): Number of samples retained for each category.
        seed (int, optional): Random seed to ensure reproducibility.

    Returns:
        torch.utils.data.Subset: A subset dataset containing few-shot samples.
    """
    if seed is not None:
        random.seed(seed)
    # Check if the dataset has a 'targets' attribute
    if not hasattr(dataset, "targets"):
        raise AttributeError(
            "The dataset object needs to have a 'targets' attribute to access the labels."
        )

    targets = dataset.targets

    # If targets is Tensor, convert to list
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    selected_indices = []
    for cls, indices in class_to_indices.items():
        if len(indices) < n_shot:
            raise ValueError(f"Class {cls} has fewer samples than {n_shot}.")
        selected = random.sample(indices, n_shot)
        selected_indices.extend(selected)

    random.shuffle(selected_indices)

    return CustomSubset(dataset, selected_indices)


from torch.utils.data import DataLoader


def calculate_prototypes(backbone, dataset, batch_size=32, device="cuda"):
    backbone.eval()
    backbone.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeds_sum = defaultdict(lambda: 0)
    embeds_count = defaultdict(lambda: 0)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # import pdb; pdb.set_trace()
            features, embeds = backbone(
                images, labels
            )  # shape: (batch_size, feature_dim=100)

            for embed, label in zip(embeds, labels):
                embeds_sum[label.item()] += embed
                embeds_count[label.item()] += 1

    # Calculate the prototype of each category
    prototypes = {}
    for label, feature_sum in embeds_sum.items():
        prototypes[label] = (feature_sum / embeds_count[label]).cpu()

    return prototypes


def replace_ffn_with_lora(model, rank=8):
    """
    Replace nn.Linear with lora.Linear in the FFN layer of the Transformer Block in the ViT model.

        parameter:
            model (torch.nn.Module): ViT model (such as torchvision.models.vit_b_16 instance).
            rank (int): Rank of LoRA layer.

        return:
            torch.nn.Module: modified model.
    """
    for name, module in model.named_modules():
        # Make sure we only modify the FFN module in the Transformer Encoder Block
        if hasattr(module, "mlp"):  # Usually FFN is in the `mlp` attribute
            ffn = module.mlp
            for ffn_name, ffn_layer in ffn.named_children():
                if isinstance(ffn_layer, nn.Linear):
                    # Replace with lora.Linear
                    in_features = ffn_layer.in_features
                    out_features = ffn_layer.out_features
                    bias = ffn_layer.bias is not None
                    lora_layer = lora.Linear(in_features, out_features, r=rank)
                    setattr(ffn, ffn_name, lora_layer)

    return model


def modify_head(model, current_id_to_original_id, device):
    """
    Modify the model's classification header to adapt it to the new dataset.
    """
    # Get the original classification head and weight
    old_classifier = model.heads.head
    old_weight = (
        old_classifier.weight.data
    )  # Weight of the original classification head
    old_bias = old_classifier.bias.data  # Offset of original classification head

    # Number of categories for the new category head
    new_num_classes = len(current_id_to_original_id)

    # Initialize new classification head
    new_classifier = nn.Linear(old_classifier.in_features, new_num_classes)

    # Extract the weights and biases of the corresponding categories from the old weights
    new_weight = torch.stack(
        [old_weight[imagenet_id] for imagenet_id in current_id_to_original_id.values()]
    )
    new_bias = torch.tensor(
        [old_bias[imagenet_id] for imagenet_id in current_id_to_original_id.values()]
    )

    # Assign the extracted weights and biases to the new classification head
    new_classifier.weight.data = new_weight
    new_classifier.bias.data = new_bias

    # Replace the model's classification head
    model.heads.head = new_classifier
    model = model.to(device)

    return model


if __name__ == "__main__":
    vers = get_val_data(data_path="./eval/", targets=["lfw"])
    embed()
