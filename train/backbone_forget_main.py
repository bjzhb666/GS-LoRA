import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb

from config import get_config
from image_iter import FaceDataset

from util.utils import (
    separate_irse_bn_paras,
    separate_resnet_bn_paras,
    separate_mobilefacenet_bn_paras,
)
from util.utils import (
    get_val_data,
    perform_val,
    get_time,
    buffer_val,
    AverageMeter,
    train_accuracy,
)
from util.utils import split_dataset

import time
from vit_pytorch_face import ViT_face
from vit_pytorch_face import ViTs_face

# from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import loralib as lora
from torch.utils.data import Subset
import numpy as np
from engine import eval_data


def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="for face verification")
    parser.add_argument(
        "-w", "--workers_id", help="gpu ids or cpu", default="cpu", type=str
    )
    parser.add_argument("-e", "--epochs", help="training epochs", default=125, type=int)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument(
        "-d",
        "--data_mode",
        help="use which database, [casia, vgg, ms1m, retina, ms1mr,casia100,casia1000]",
        default="casia100",
        type=str,
    )
    parser.add_argument(
        "-n", "--net", help="which network, ['VIT','VITs']", default="VITs", type=str
    )
    parser.add_argument(
        "-head",
        "--head",
        help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']",
        default="ArcFace",
        type=str,
    )
    parser.add_argument("-r", "--resume", help="resume model", default="", type=str)
    parser.add_argument("--outdir", help="output dir", default="", type=str)

    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        metavar="N",
        help="dataloader threads (default: 4)",
    )

    # lora rank on FFN of Transformer blocks
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        metavar="N",
        help="lora rank on FFN of Transformer blocks (default: 8)",
    )
    # wandb offline
    parser.add_argument(
        "--wandb_offline",
        default=False,
        action="store_true",
    )
    # VIT depth
    parser.add_argument(
        "--vit_depth", type=int, default=10, metavar="N", help="vit depth (default: 20)"
    )
    # forget and remain cls
    parser.add_argument(
        "--num_of_first_cls",
        type=int,
        default=50,
        metavar="N",
        help="number of first cls (default: 50)",
    )
    parser.add_argument(
        "--per_forget_cls",
        type=int,
        default=50,
        metavar="N",
        help="number of forget cls (default: 50)",
    )
    parser.add_argument("--wandb_group", type=str, default="casia100")

    parser.add_argument("--forget_data_ratio", type=float, default=0.1)
    parser.add_argument("--remain_data_ratio", type=float, default=0.1)
    args = parser.parse_args()

    # ======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg["SEED"]  # random seed for reproduce results
    torch.manual_seed(SEED)
    random_generator = torch.Generator().manual_seed(SEED)

    DATA_ROOT = cfg[
        "DATA_ROOT"
    ]  # the parent root where your train/val/test data are stored
    EVAL_PATH = cfg["EVAL_PATH"]
    WORK_PATH = cfg[
        "WORK_PATH"
    ]  # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg[
        "BACKBONE_RESUME_ROOT"
    ]  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg["BACKBONE_NAME"]
    HEAD_NAME = cfg[
        "HEAD_NAME"
    ]  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']

    INPUT_SIZE = cfg["INPUT_SIZE"]
    EMBEDDING_SIZE = cfg["EMBEDDING_SIZE"]  # feature dimension
    BATCH_SIZE = cfg["BATCH_SIZE"]
    NUM_EPOCH = cfg["NUM_EPOCH"]

    DEVICE = cfg["DEVICE"]
    MULTI_GPU = cfg["MULTI_GPU"]  # flag to use multiple GPUs
    GPU_ID = cfg["GPU_ID"]  # specify your GPU ids
    print("GPU_ID", GPU_ID)
    WORKERS = cfg["WORKERS"]
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, "config.txt"), "w") as f:
        f.write(str(cfg))
    with open(os.path.join(WORK_PATH, "args.txt"), "w") as f:
        f.write(str(args))
    print("=" * 60)

    wandb.init(
        project="face recognition",
        group=args.wandb_group,
        mode="offline" if args.wandb_offline else "online",
    )
    wandb.config.update(args)
    # writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True

    # with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
    #     NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    if args.data_mode == "casia100":
        NUM_CLASS = 100
    elif args.data_mode == "casia1000":
        NUM_CLASS = 1000
    h, w = 112, 112

    assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    order_list = [
        83,
        17,
        10,
        9,
        89,
        52,
        7,
        32,
        37,
        77,
        61,
        34,
        78,
        55,
        56,
        63,
        69,
        22,
        75,
        66,
        35,
        72,
        71,
        23,
        86,
        38,
        24,
        5,
        29,
        30,
        16,
        11,
        82,
        19,
        57,
        14,
        91,
        27,
        98,
        53,
        0,
        20,
        59,
        28,
        47,
        18,
        6,
        48,
        12,
        62,
        1,
        43,
        3,
        60,
        41,
        85,
        4,
        95,
        97,
        36,
        94,
        65,
        67,
        99,
        25,
        33,
        70,
        40,
        92,
        31,
        58,
        15,
        45,
        2,
        87,
        76,
        80,
        44,
        64,
        8,
        51,
        54,
        13,
        88,
        84,
        26,
        50,
        39,
        96,
        81,
        49,
        42,
        21,
        93,
        74,
        73,
        46,
        90,
        68,
        79,
    ]
    # get forget and remain dataset
    # split datasets
    # 1. calculate st1, en1, st2, en2
    st1 = 0
    en1 = args.num_of_first_cls
    st2 = en1
    en2 = en1 + args.per_forget_cls
    # 2. split datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_ROOT, "train"), transform=data_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_ROOT, "test"), transform=data_transform
    )
    remain_dataset_train, forget_dataset_train = split_dataset(
        dataset=train_dataset,
        class_order_list=order_list,
        split1_start=st1,
        split1_end=en1,
        split2_start=st2,
        split2_end=en2,
    )
    remain_dataset_test, forget_dataset_test = split_dataset(
        dataset=test_dataset,
        class_order_list=order_list,
        split1_start=st1,
        split1_end=en1,
        split2_start=st2,
        split2_end=en2,
    )

    # get sub datasets
    len_forget_dataset_train = len(forget_dataset_train)
    len_remain_dataset_train = len(remain_dataset_train)
    subset_size_forget = int(len_forget_dataset_train * args.forget_data_ratio)
    subset_size_remain = int(len_remain_dataset_train * args.remain_data_ratio)

    subset_indices_forget = torch.randperm(len_forget_dataset_train)[
        :subset_size_forget
    ]
    subset_indices_remain = torch.randperm(len_remain_dataset_train)[
        :subset_size_remain
    ]

    forget_dataset_train_sub = Subset(forget_dataset_train, subset_indices_forget)
    remain_dataset_train_sub = Subset(remain_dataset_train, subset_indices_remain)

    combined_dataset_train = torch.utils.data.ConcatDataset(
        [forget_dataset_train_sub, remain_dataset_train_sub]
    )

    # get dataloader
    train_loader_forget = torch.utils.data.DataLoader(
        forget_dataset_train_sub,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
    )
    train_loader_remain = torch.utils.data.DataLoader(
        remain_dataset_train_sub,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
    )
    testloader_forget = torch.utils.data.DataLoader(
        forget_dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
    )
    testloader_remain = torch.utils.data.DataLoader(
        remain_dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
    )
    combined_loader_train = torch.utils.data.DataLoader(
        combined_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
    )

    print("len(train_loader_forget)", len(train_loader_forget))
    print("len(train_loader_remain)", len(train_loader_remain))
    print("len(testloader_forget)", len(testloader_forget))
    print("len(testloader_remain)", len(testloader_remain))
    print("len(combined_loader_train)", len(combined_loader_train))
    # import pdb; pdb.set_trace()
    # testloader = torch.utils.data.DataLoader(test_dataset,
    #                                          batch_size=BATCH_SIZE,
    #                                          shuffle=False,
    #                                          num_workers=WORKERS,
    #                                          drop_last=False)

    print("Number of Training Classes: {}".format(NUM_CLASS))

    # # dataset = FaceDataset(os.path.join(DATA_ROOT, 'train.rec'), rand_mirror=True)
    # dataset = datasets.ImageFolder(root=DATA_ROOT, transform=data_transform)

    # train_dataset = datasets.ImageFolder(root=os.path.join(DATA_ROOT, 'train'),transform=data_transform)
    # test_dataset = datasets.ImageFolder(root=os.path.join(DATA_ROOT, 'test'),transform=data_transform)

    # trainloader = torch.utils.data.DataLoader(train_dataset,
    #                                           batch_size=BATCH_SIZE,
    #                                           shuffle=True,
    #                                           num_workers=WORKERS,
    #                                           drop_last=True)
    # testloader = torch.utils.data.DataLoader(test_dataset,
    #                                          batch_size=BATCH_SIZE,
    #                                          shuffle=False,
    #                                          num_workers=WORKERS,
    #                                          drop_last=False)

    # print("Number of Training Classes: {}".format(NUM_CLASS))

    highest_acc = 0.0

    # embed()
    # ======= model & loss & optimizer =======#
    BACKBONE_DICT = {
        "VIT": ViT_face(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            dim=512,
            depth=args.vit_depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank,
        ),
        "VITs": ViTs_face(
            loss_type=HEAD_NAME,
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=args.vit_depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank,
        ),
    }
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    LOSS = nn.CrossEntropyLoss()

    # embed()
    OPTIMIZER = create_optimizer(args, BACKBONE)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)
    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT), strict=False)
        else:
            print(
                "No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(
                    BACKBONE_RESUME_ROOT
                )
            )
        print("=" * 60)

    # only train the last layer
    for name, param in BACKBONE.named_parameters():
        if "loss" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 统计BACKBONE的可训练参数量
    learnable_parameters = count_trainable_parameters(BACKBONE)
    print("learnable_parameters", learnable_parameters)  # 31760896
    # print("ratio of learnable_parameters", learnable_parameters/31760896) # 0.011938
    wandb.log(
        {
            "learnable_parameters": learnable_parameters,
            #    'ratio of learnable_parameters': learnable_parameters/31760896,
            "lora_rank": args.lora_rank,
        }
    )
    # exit()

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    # ======= train & validation & save checkpoint =======#
    DISP_FREQ = 5  # frequency to display training loss & acc
    VER_FREQ = 5  # frequency to perform validation

    batch = 0  # batch index

    losses = AverageMeter()
    top1 = AverageMeter()

    BACKBONE.train()  # set to training mode
    forget_acc = []
    remain_acc = []

    # eval before training
    print("Perform Evaluation on forget test set and remain test set...")
    forget_acc_before = eval_data(BACKBONE, testloader_forget, DEVICE, "forget", batch)
    remain_acc_before = eval_data(BACKBONE, testloader_remain, DEVICE, "remain", batch)
    wandb.log(
        {"forget_acc_before": forget_acc_before, "remain_acc_before": remain_acc_before}
    )

    for epoch in range(NUM_EPOCH):  # start training process

        lr_scheduler.step(epoch)

        last_time = time.time()

        for inputs, labels in iter(combined_loader_train):

            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            # print("inputs", inputs.shape, inputs.dtype)
            # print("labels", labels.shape, labels.dtype)
            # print("labels", labels)
            outputs, emb = BACKBONE(inputs.float(), labels)
            loss = LOSS(outputs, labels)

            # print("outputs", outputs, outputs.data)
            # measure accuracy and record loss
            prec1 = train_accuracy(outputs.data, labels, topk=(1,))

            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                epoch_loss = losses.avg
                epoch_acc = top1.avg
                # writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                # writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)
                wandb.log(
                    {
                        "Training/Training_Loss": epoch_loss,
                        "Training/Training_Accuracy": epoch_acc,
                    },
                    step=batch + 1,
                )
                batch_time = time.time() - last_time
                last_time = time.time()

                print(
                    "Epoch {} Batch {}\t"
                    "Speed: {speed:.2f} samples/s\t"
                    "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch + 1,
                        batch + 1,
                        speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                        loss=losses,
                        top1=top1,
                    )
                )
                # print("=" * 60)
                losses = AverageMeter()
                top1 = AverageMeter()

            if (
                (batch + 1) % VER_FREQ == 0
            ) and batch != 0:  # perform validation & save checkpoints (buffer for visualization)
                for params in OPTIMIZER.param_groups:
                    lr = params["lr"]
                    break
                print("Learning rate %f" % lr)
                print("Perform Evaluation on test set and Save Checkpoints...")

                BACKBONE.eval()  # set to evaluation mode
                # 遍历测试集
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in testloader_forget:
                        # 在这里进行测试操作
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE).long()

                        outputs, _ = BACKBONE(images, labels)  # 假设model是你的模型
                        # import pdb; pdb.set_trace()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                # 打印测试精度
                accuracy = 100 * correct / total
                print("Test forget Accuracy: {:.2f}%".format(accuracy))
                wandb.log({"Test forget Accuracy": accuracy}, step=batch + 1)
                forget_acc.append(accuracy)

                BACKBONE.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in testloader_remain:
                        # 在这里进行测试操作
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE).long()

                        outputs, _ = BACKBONE(images, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                # 打印测试精度
                accuracy = 100 * correct / total
                print("Test remain Accuracy: {:.2f}%".format(accuracy))
                wandb.log({"Test remain Accuracy": accuracy}, step=batch + 1)
                remain_acc.append(accuracy)

                BACKBONE.train()  # set to training mode

            batch += 1  # batch index

    # save forget and remain acc
    forget_acc = np.array(forget_acc)
    remain_acc = np.array(remain_acc)
    np.save(os.path.join(WORK_PATH, "forget_acc.npy"), forget_acc)
    np.save(os.path.join(WORK_PATH, "remain_acc.npy"), remain_acc)

    wandb.run.name = (
        args.net
        + args.data_mode
        + args.head
        + str(args.lr)
        + "bs"
        + str(args.batch_size)
        + "ep"
        + str(args.epochs)
    )
