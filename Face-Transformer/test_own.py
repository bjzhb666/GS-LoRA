import torch
import torch.nn as nn
import sys
from vit_pytorch_face import ViT_face
from vit_pytorch_face import ViTs_face
from util.utils import get_val_data, perform_val
from IPython import embed
import sklearn
import cv2
import numpy as np
from image_iter import FaceDataset
import torch.utils.data as data
import argparse
import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb

def main(args):
    print(args)
    MULTI_GPU = False
    # set device
    GPU_ID = [int(i) for i in args.workers_id.split(',')]
    DEVICE = torch.device('cuda:%d' % GPU_ID[0]) 
    # DATA_ROOT = '/raid/Data/ms1m-retinaface-t1/'
    # with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
    #     NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    NUM_CLASS = 100 # CASIA-WebFace-sub100
    if args.network == 'VIT' :
        model = ViT_face(
            image_size=112,
            patch_size=8,
            loss_type='CosFace',
            GPU_ID= GPU_ID,
            num_class=NUM_CLASS,
            dim=512,
            depth=args.depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank,
            lora_pos=args.lora_pos
        )
    elif args.network == 'VITs':
        model = ViTs_face(
            loss_type='CosFace',
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=args.depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank
        )

    model_root = args.model
    model.load_state_dict(torch.load(model_root))

    w = torch.load(model_root)
    for x in w.keys():
        print(x, w[x].shape)
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(root='./data/faces_webface_112x112_sub100_train_test/test',transform=data_transform)    
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             drop_last=False)


    model.to(DEVICE)
    model.eval()
    # 遍历测试集
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            # 在这里进行测试操作
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()

            outputs, _ = model(images, labels)  # 假设model是你的模型
            # import pdb; pdb.set_trace()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 打印测试精度
    accuracy = 100 * correct / total
    print('\n')
    print('Test Accuracy: {:.2f}%'.format(accuracy))
    print('\n')
    # wandb.log({"Test Accuracy": accuracy})


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-800-150de-depth6/Backbone_VIT_Epoch_716_Batch_52980_Time_2023-10-14-12-35_checkpoint.pth', help='pretrained model')
    parser.add_argument('--network', default='VIT',
                        help='training set directory')
    parser.add_argument('--batch_size', type=int, help='', default=20)
    parser.add_argument('--lora_rank', type=int, help='', default=0)
     # lora pos (FFN and attention) on Transformer blocks
    parser.add_argument(
        '--lora_pos',
        type=str,
        default='FFN',
        help='lora pos (FFN and attention) on Transformer blocks (default: FFN)')
    parser.add_argument('--depth', type=int, help='', default=6)
    parser.add_argument('--num_workers', type=int, help='', default=4)
    parser.add_argument("-w",
                        "--workers_id",
                        help="gpu ids or cpu",
                        default='cpu',
                        type=str)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))