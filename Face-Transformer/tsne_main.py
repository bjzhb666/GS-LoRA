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
from sklearn.metrics import classification_report
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt_sne
from sklearn.manifold import TSNE
import os
from IPython import embed

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
            loss_type=args.head,
            GPU_ID= GPU_ID,
            num_class=NUM_CLASS,
            dim=512,
            depth=args.depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank
        )
    elif args.network == 'VITs':
        model = ViTs_face(
            loss_type=args.head,
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
    # data_root='./data/faces_webface_112x112_sub100_train_test/test'
    data_root='./data/faces_Tsne_sub'
    test_dataset = datasets.ImageFolder(root=data_root,transform=data_transform)    
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             drop_last=False)

    print(test_dataset.class_to_idx)
    # {'138': 0, '6332': 1, '6558': 2, '658': 3, '6811': 4, '819': 5, '944': 6}
 
    model.eval()
    # 遍历测试集
    model.to(DEVICE)
    correct = 0
    total = 0

    # 存储所有类别的预测outputs和标签
    all_predicted = []
    all_labels = []
    all_outputs = []
    all_embeds = []
   
    with torch.no_grad():
        for images, labels in testloader:
            # 在这里进行测试操作
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()
            # import pdb; pdb.set_trace()
            outputs, embed = model(images, labels)  # 假设model是你的模型
            # import pdb; pdb.set_trace()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 存储所有类别的预测outputs和标签
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            all_embeds.extend(embed.cpu().numpy())
            
        # 把all_predicted和all_labels转换成numpy array
        all_predicted = np.array(all_predicted)
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)
        all_embeds = np.array(all_embeds)
        # embed()
    # 打印测试精度
    accuracy = 100 * correct / total
    print('\n')
    print('Test Accuracy: {:.2f}%'.format(accuracy))
    print('\n')
    # wandb.log({"Test Accuracy": accuracy})
    # class_report = classification_report(all_labels, np.argmax(all_outputs, axis=1))
    # print(class_report)
    if args.mode=="before":
        name="before"
    elif args.mode=="after":
        name="after"
    plot_tsne(all_outputs, all_labels, name, fileNameDir="test-Tsne",mode=args.mode)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/zhaohongbo/Github/amnesic-face-recognition/Face-Transformer/results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6-new/Backbone_VIT_Epoch_1110_Batch_82100_Time_2023-10-18-18-22_checkpoint.pth', help='pretrained model')
    parser.add_argument('--network', default='VIT',
                        help='training set directory')
    parser.add_argument('--batch_size', type=int, help='', default=20)
    parser.add_argument('--lora_rank', type=int, help='', default=0)
    parser.add_argument('--depth', type=int, help='', default=6)
    parser.add_argument('--num_workers', type=int, help='', default=4)
    parser.add_argument("-w",
                        "--workers_id",
                        help="gpu ids or cpu",
                        default='cpu',
                        type=str)
    parser.add_argument(
        "-head",
        "--head",
        help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']",
        default='ArcFace',
        type=str)
    parser.add_argument('--mode', type=str)
    return parser.parse_args(argv)

def plot_tsne(features, labels, epoch,fileNameDir = None, mode=None):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    # 创建目标文件夹
    if not os.path.exists(fileNameDir):
        os.makedirs(fileNameDir)
    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    import seaborn as sns
 
    #查看标签的种类有几个
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
 
    try:
        tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    except:
        tsne_features = tsne.fit_transform(features)
 
    x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    tsne_features = (tsne_features - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    #一个类似于表格的数据结构
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]
 
 
    # 颜色是根据标签的大小顺序进行赋色.
    hex = ["#e45a82", "#82e45a", "#5a82e4", "#fdb157", "#57fdb1", "#b157fd", "#fd5a57"] # 绿、红
    data_label = []
    for v in df.y.tolist():
        if v == 0:
            data_label.append("f1")
        elif v == 1:
            data_label.append("r1")
        elif v == 2:
            data_label.append("r2")
        elif v == 3:
            data_label.append("r3")
        elif v == 4:
            data_label.append("f2")
        elif v == 5:
            data_label.append("r4")
        elif v == 6:
            data_label.append("r5")

    df["value"] = data_label
    
    if mode=="before":
        title = "Before Forgetting"
    elif mode=="after":
        title = "After Forgetting"
    # hue=df.y.tolist()
    # hue:根据y列上的数据种类，来生成不同的颜色；
    # style:根据y列上的数据种类，来生成不同的形状点；
    # s:指定显示形状的大小
    sns.scatterplot(x= df.comp1.tolist(), y= df.comp2.tolist(),hue=df.value.tolist(),style = df.value.tolist(),
                    palette=sns.color_palette(hex,class_num),markers= {"r1":".","r2":".","r3":".","r4":".","r5":".","f1":",","f2":","},
                    # s = 10,
                    data=df).set(title=title) #T-SNE projection
  
 
   
    # 指定图注的位置 "lower right"
    plt_sne.legend(loc = "lower right")
    # 不要坐标轴
    plt_sne.axis("off")
    # 保存图像
    plt_sne.savefig(os.path.join(fileNameDir,"%s.jpg") % str(epoch),format = "jpg",dpi = 300)
    # plt_sne.show()



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))