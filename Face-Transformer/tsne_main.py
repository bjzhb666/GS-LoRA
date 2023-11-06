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

def transfer_label(labels, original_class_to_idx, new_class_to_idx):
    # labels: tensor, shape: (batch_size, )
    # original_class_to_idx: dict, eg: {'138': 0, '6332': 1, '6558': 2, '658': 3, '6811': 4, '819': 5, '944': 6}
    # new_class_to_idx: dict, eg: {'138': 0, '6332': 1, '6558': 2, '658': 3, '6811': 4, '819': 5, '944': 6}
    # return: tensor, shape: (batch_size, )
    list_labels = labels.numpy().astype(int).tolist()
    new_labels = []

    for value in list_labels:
        # 遍历第一个字典，找到对应的key
        for key, val in new_class_to_idx.items():
            if val == value:
                # 根据key获取第二个字典的值
                new_label = original_class_to_idx[int(key)]
                new_labels.append(new_label)
                break
    
    new_labels = torch.tensor(new_labels)
    return new_labels

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
    model.load_state_dict(torch.load(model_root),strict=False)

    w = torch.load(model_root)
    for x in w.keys():
        print(x, w[x].shape)
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # data_root='./data/faces_webface_112x112_sub100_train_test/test'
    data_root='./data/faces_Tsne_sub/test'
    test_dataset = datasets.ImageFolder(root=data_root,transform=data_transform)    
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             drop_last=False)

    print(test_dataset.class_to_idx)
    new_class_to_idx = test_dataset.class_to_idx
    # {'138': 0, '6332': 1, '6558': 2, '658': 3, '6811': 4, '819': 5, '944': 6}
    original_class_to_idx = {1005: 0,
        1014: 1,
        1022: 2,
        1039: 3,
        1147: 4,
        1154: 5,
        1226: 6,
        1236: 7,
        1245: 8,
        1248: 9,
        1299: 10,
        1317: 11,
        1321: 12,
        138: 13,
        14: 14,
        1466: 15,
        1542: 16,
        1595: 17,
        1649: 18,
        1657: 19,
        176: 20,
        1764: 21,
        18: 22,
        1822: 23,
        194: 24,
        199: 25,
        1994: 26,
        2013: 27,
        2027: 28,
        2059: 29,
        21: 30,
        2188: 31,
        2189: 32,
        23: 33,
        2390: 34,
        2439: 35,
        2526: 36,
        256: 37,
        2639: 38,
        2644: 39,
        2725: 40,
        2738: 41,
        2749: 42,
        2834: 43,
        291: 44,
        2986: 45,
        3: 46,
        3024: 47,
        308: 48,
        3274: 49,
        3384: 50,
        3410: 51,
        343: 52,
        3687: 53,
        37: 54,
        3989: 55,
        406: 56,
        41: 57,
        4111: 58,
        4238: 59,
        4290: 60,
        4312: 61,
        4362: 62,
        4401: 63,
        4478: 64,
        4670: 65,
        47: 66,
        476: 67,
        4968: 68,
        5065: 69,
        513: 70,
        5287: 71,
        5444: 72,
        5447: 73,
        5547: 74,
        578: 75,
        5830: 76,
        5889: 77,
        6023: 78,
        6049: 79,
        6090: 80,
        6224: 81,
        626: 82,
        6332: 83,
        6374: 84,
        6558: 85,
        658: 86,
        6723: 87,
        6811: 88,
        69: 89,
        693: 90,
        7601: 91,
        7766: 92,
        8071: 93,
        819: 94,
        8313: 95,
        8401: 96,
        872: 97,
        944: 98,
        956: 99
    }
    

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
            maped_labels = transfer_label(labels, original_class_to_idx, new_class_to_idx)    
            labels = maped_labels.to(DEVICE).long()
            
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
        # 保存all_predicted和all_labels
        np.save('all_outputs.npy', all_outputs)
        np.save('all_predicted.npy', all_predicted)
        np.save('all_labels.npy', all_labels)
        # embed()
    # 打印测试精度
    accuracy = 100 * correct / total
    print('\n')
    print('Test Accuracy: {:.2f}%'.format(accuracy))
    print('\n')

    class_report = classification_report(all_labels, np.argmax(all_outputs, axis=1))
    print(class_report)
    if args.mode=="before":
        name="before"
    elif args.mode=="after":
        name="after"
    plot_tsne(all_embeds, all_labels, name, fileNameDir="test-Tsne",mode=args.mode)
    plot_tsne(all_outputs, all_labels, name+'outputs', fileNameDir="test-Tsne",mode=args.mode)


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
    hex = ["#c957db", "#dd5f57", "#b9db57", "#57db30", "#5784db", "#f2a542", "#ed0017", "#db57a6", "#57db83", "#c3db57"] # 绿、红
    data_label = []
    # 94 98 85 86 83 57 20 19作为保留类819 944 6558 658 6332 41 176 1657
    # 13 88做遗忘类别 138 6811
    for v in df.y.tolist():
        if v == 13:
            data_label.append("f1")
        elif v == 94:
            data_label.append("r1")
        elif v == 98:
            data_label.append("r2")
        elif v == 85:
            data_label.append("r3")
        elif v == 88:
            data_label.append("f2")
        elif v == 83:
            data_label.append("r4")
        elif v == 86:
            data_label.append("r5")
        elif v == 57:
            data_label.append("r6")
        elif v == 20:
            data_label.append("r7")
        elif v == 19:
            data_label.append("r8")

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
                    palette=sns.color_palette(hex,class_num),
                    markers= {"r1":".","r2":".","r3":".","r4":".","r5":".","r6":".","r7":".","r8":".","f1":"*","f2":"*"},
                    # s = 10,
                    data=df).set(title=title) #T-SNE projection
  
 
   
    # 指定图注的位置 "lower right"
    plt_sne.legend(loc=(0.97,0.05))
    # 不要坐标轴
    plt_sne.axis("off")
    # 保存图像
    plt_sne.savefig(os.path.join(fileNameDir,"%s.jpg") % str(epoch),format = "jpg",dpi = 300)
    # plt_sne.show()
    # 清除画布
    plt_sne.clf()



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))