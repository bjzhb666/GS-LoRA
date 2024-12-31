import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
import random
from config import get_config
from image_iter import CLDatasetWrapper, CustomSubset

from util.utils import (
    get_unique_classes,
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
from engine import train_one_epoch, eval_data
from engine_cl import train_one_epoch_regularzation
from torch.utils.data import Subset

from IPython import embed

from util.cal_norm import get_norm_of_lora
from torch.utils.data import DataLoader


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
        help="use which database, [casia100, casia1000]",
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
        default=4,
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
        "--vit_depth", type=int, default=6, metavar="N", help="vit depth (default: 6)"
    )

    # add forget parameters
    parser.add_argument(
        "--num_of_first_cls", type=int, default=90, help="number of first class"
    )
    parser.add_argument("--per_forget_cls", type=int, default=10)
    parser.add_argument("--BND", type=float, default=10)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.1)

    # mode selection
    parser.add_argument(
        "--one_stage",
        default=True,
        action="store_false",
        help="whether to use one stage training",
    )
    parser.add_argument(
        "--l2", default=False, action="store_true", help="whether to use l2 norm"
    )
    parser.add_argument(
        "--l2_lambda", default=0.1, type=float, help="lambda for l2 norm"
    )
    parser.add_argument(
        "--ewc", default=False, action="store_true", help="whether to use ewc"
    )
    parser.add_argument("--ewc_lambda", default=0.1, type=float, help="lambda for ewc")
    parser.add_argument(
        "--MAS", default=False, action="store_true", help="whether to use mas"
    )
    parser.add_argument("--mas_lambda", default=0.1, type=float, help="lambda for mas")
    parser.add_argument(
        "--si", default=False, action="store_true", help="whether to use si"
    )
    parser.add_argument("--si_c", default=0.1, type=float, help="c for si")
    parser.add_argument(
        "--online", default=False, action="store_true", help="whether to use online"
    )
    parser.add_argument(
        "--replay", default=False, action="store_true", help="whether to use replay"
    )
    parser.add_argument(
        "--retrain", default=False, action="store_true", help="whether to retrain"
    )
    parser.add_argument(
        "--n_fisher_sample", default=None, type=int, help="number of fisher sample"
    )
    # wramup for alpha
    parser.add_argument(
        "--warmup_alpha",
        default=False,
        action="store_true",
        help="whether to use warmup_alpha",
    )
    parser.add_argument(
        "--big_alpha", default=0, type=float, help="big alpha for warmup_alpha"
    )
    parser.add_argument("--alpha_epoch", default=20, type=int, help="epoch for alpha")
    # beta decay
    parser.add_argument(
        "--beta_decay",
        default=False,
        action="store_true",
        help="whether to use beta_decay",
    )
    parser.add_argument(
        "--small_beta", default=1e-4, type=float, help="small beta for beta_decay"
    )
    parser.add_argument(
        "--wandb_group", default=None, type=str, help="wandb group name"
    )
    # grouping strategy
    parser.add_argument(
        "--grouping", default="block", type=str, help="grouping strategy"
    )
    # data ratio
    parser.add_argument("--data_ratio", default=0.1, type=float, help="data ratio")
    # open class number
    parser.add_argument(
        "--open_cls_num", default=10, type=int, help="open class number"
    )

    # output FFN freeze
    parser.add_argument(
        "--ffn_open", default=False, action="store_true", help="whether to freeze ffn"
    )
    args = parser.parse_args()

    # ======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg["SEED"]  # random seed for reproduce results
    torch.manual_seed(SEED)
    import numpy as np

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    # dataset = FaceDataset(os.path.join(DATA_ROOT, 'train.rec'), rand_mirror=True)
    # dataset = datasets.ImageFolder(root=DATA_ROOT, transform=data_transform)

    # create order list
    order_list = list(range(NUM_CLASS))
    # shuffle order list

    random.seed(SEED)
    random.shuffle(order_list)
    print("order_list", order_list)

    # split datasets
    # 1. calculate st1, en1, st2, en2
    st1 = 0
    en1 = args.num_of_first_cls - args.open_cls_num  # 90-10
    # for open setting
    st3 = en1  # 90-10
    en3 = args.num_of_first_cls  # 90
    st2 = en3  # 90
    en2 = en3 + args.per_forget_cls  # 90+10

    print("st1, en1, st2, en2, st3, en3", st1, en1, st2, en2, st3, en3)
    # 2. split datasets:
    #  - forget set
    #  - remain set (classes should be retained with replay),
    #  - open set (classes should be retrained without replay)
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_ROOT, "train"), transform=data_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_ROOT, "test"), transform=data_transform
    )
    remain_dataset_train, open_dataset_train = split_dataset(
        dataset=train_dataset,
        class_order_list=order_list,
        split1_start=st1,
        split1_end=en1,
        split2_start=st3,
        split2_end=en3,
    )
    # remain_dataset_train, forget_dataset_train = split_dataset(dataset=train_dataset,
    #                                                            class_order_list=order_list,
    #                                                            split1_start=st1,
    #                                                            split1_end=en1,
    #                                                            split2_start=st2,
    #                                                            split2_end=en2)
    _, forget_dataset_train = split_dataset(
        dataset=train_dataset,
        class_order_list=order_list,
        split1_start=st1,
        split1_end=en3,
        split2_start=st2,
        split2_end=en2,
    )
    remain_dataset_test, open_dataset_test = split_dataset(
        dataset=test_dataset,
        class_order_list=order_list,
        split1_start=st1,
        split1_end=en1,
        split2_start=st3,
        split2_end=en3,
    )
    _, forget_dataset_test = split_dataset(
        dataset=test_dataset,
        class_order_list=order_list,
        split1_start=st1,
        split1_end=en3,
        split2_start=st2,
        split2_end=en2,
    )

    num_remain_classes, num_remain_total = get_unique_classes(
        remain_dataset_train, train_dataset
    )
    num_forget_classes, num_forget_total = get_unique_classes(
        forget_dataset_train, train_dataset
    )
    print(num_forget_classes, len(num_forget_classes))
    print(num_remain_classes, len(num_remain_classes))
    print("Same class?", bool(set(num_remain_classes) & set(num_forget_classes)))

    # remain_dataset_test, forget_dataset_test = split_dataset(dataset=test_dataset,
    #                                                             class_order_list=order_list,
    #                                                             split1_start=st1,
    #                                                             split1_end=en1,
    #                                                             split2_start=st2,
    #                                                             split2_end=en2)
    # _,open_dataset_test = split_dataset(dataset=test_dataset,
    #                                                             class_order_list=order_list,
    #                                                             split1_start=st1,
    #                                                             split1_end=en2,
    #                                                             split2_start=st3,
    #                                                             split2_end=en3)

    # get sub datasets
    len_forget_dataset_train = len(forget_dataset_train)
    len_remain_dataset_train = len(remain_dataset_train)
    print("len_forget_dataset_train", len_forget_dataset_train)
    print("len_remain_dataset_train", len_remain_dataset_train)
    subset_size_forget = int(len_forget_dataset_train * args.data_ratio)
    subset_size_remain = int(len_remain_dataset_train * args.data_ratio)

    subset_indices_forget = torch.randperm(len_forget_dataset_train)[
        :subset_size_forget
    ]
    subset_indices_remain = torch.randperm(len_remain_dataset_train)[
        :subset_size_remain
    ]

    forget_dataset_train_sub = CustomSubset(forget_dataset_train, subset_indices_forget)
    remain_dataset_train_sub = CustomSubset(remain_dataset_train, subset_indices_remain)

    # get remain all datasets
    remain_dataset_test_all = torch.utils.data.ConcatDataset(
        [remain_dataset_test, open_dataset_test]
    )

    if not args.one_stage:
        forget_dataset_train_sub = CLDatasetWrapper(forget_dataset_train_sub)
        # get total available training datasets
        print("\033[33mConcat forget datsets and remain datasets\033[0m")
        total_dataset_train = torch.utils.data.ConcatDataset(
            [forget_dataset_train_sub, remain_dataset_train_sub]
        )
    forget_train_generator = torch.Generator()
    forget_train_generator.manual_seed(SEED)
    remain_train_generator = torch.Generator()
    remain_train_generator.manual_seed(SEED)
    total_train_generator = torch.Generator()
    total_train_generator.manual_seed(SEED)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)

    train_loader_forget = torch.utils.data.DataLoader(
        forget_dataset_train_sub,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=forget_train_generator,
    )
    train_loader_remain = torch.utils.data.DataLoader(
        remain_dataset_train_sub,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=remain_train_generator,
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
    testloader_open = torch.utils.data.DataLoader(
        open_dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
    )
    testloader_remain_all = torch.utils.data.DataLoader(
        remain_dataset_test_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
    )

    train_loader_total = torch.utils.data.DataLoader(
        total_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=total_train_generator,
    )

    print("len(train_loader_forget)", len(train_loader_forget))
    print("len(train_loader_remain)", len(train_loader_remain))
    print("len(testloader_forget)", len(testloader_forget))
    print("len(testloader_remain)", len(testloader_remain))
    print("len(testloader_open)", len(testloader_open))
    print("len(testloader_remain_all)", len(testloader_remain_all))
    # import pdb; pdb.set_trace()
    # testloader = torch.utils.data.DataLoader(test_dataset,
    #                                          batch_size=BATCH_SIZE,
    #                                          shuffle=False,
    #                                          num_workers=WORKERS,
    #                                          drop_last=False)

    print("Number of Training Classes: {}".format(NUM_CLASS))

    highest_H_mean = 0.0

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

    # LOSS = LossFaceCE(type=HEAD_NAME,dim=512,num_class=NUM_CLASS, GPU_ID=GPU_ID)
    LOSS = nn.CrossEntropyLoss()
    # embed()
    OPTIMIZER = create_optimizer(args, BACKBONE)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)
    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and not args.retrain:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            missing_keys, unexpected_keys = BACKBONE.load_state_dict(
                torch.load(BACKBONE_RESUME_ROOT), strict=False
            )
            if len(missing_keys) > 0:
                print("Missing keys: {}".format(missing_keys))
                print("\n")
                for missing_key in missing_keys:
                    if "lora" not in missing_key:
                        print("\033[31mWrong resume.\033[0m")
                        exit()
            if len(unexpected_keys) > 0:
                print("Unexpected keys: {}".format(unexpected_keys))
                print("\n")
        else:
            print(
                "No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(
                    BACKBONE_RESUME_ROOT
                )
            )
        print("=" * 60)

    # Create importance dataset and dataloader for the first task in CL methods.
    import_st1 = 0
    import_en1 = args.num_of_first_cls - args.per_forget_cls  # only one task
    import_st2 = import_en1
    import_en2 = import_en1 + args.per_forget_cls
    importance_dataset_train, _ = split_dataset(
        dataset=train_dataset,
        class_order_list=order_list,
        split1_start=import_st1,
        split1_end=import_en1,
        split2_start=import_st2,
        split2_end=import_en2,
    )
    # len_importance_dataset_train = len(train_dataset)
    # subset_size_importance = int(len_importance_dataset_train * 0.1)
    # subset_indices_importance = torch.randperm(len_importance_dataset_train)[
    #     :subset_size_importance
    # ]
    # importance_dataset_train = Subset(train_dataset, subset_indices_importance)
    importance_dataloader_train = torch.utils.data.DataLoader(
        importance_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
    )
    name_importance_classes, num_importance_classes = get_unique_classes(
        importance_dataset_train, train_dataset
    )
    print("importance class", name_importance_classes, num_importance_classes)

    if args.one_stage:
        if args.lora_rank > 0:
            lora.mark_only_lora_as_trainable(BACKBONE)
            print("Use LoRA in Transformer FFN, loar_rank: ", args.lora_rank)
            # for n,p in BACKBONE.named_parameters():

        else:
            print(
                "Do not use LoRA in Transformer FFN, train all parameters."
            )  # 19,157,504
    else:  # CL baselines
        for n, p in BACKBONE.named_parameters():
            if "loss" in n and not args.ffn_open:  # Turn off the gradient of output FFN
                p.requires_grad = False

    learnable_parameters = count_trainable_parameters(BACKBONE)
    print("learnable_parameters", learnable_parameters)  # 19,157,504
    print("ratio of learnable_parameters", learnable_parameters / 19157504)
    wandb.log(
        {
            "learnable_parameters": learnable_parameters,
            "ratio of learnable_parameters": learnable_parameters / 19157504,
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

    batch = 0  # batch index
    if args.one_stage:
        losses_forget = AverageMeter()
        top1_forget = AverageMeter()
        losses_remain = AverageMeter()
        top1_remain = AverageMeter()
        losses_total = AverageMeter()
        losses_structure = AverageMeter()
    else:
        losses_CE = AverageMeter()
        losses_reg = AverageMeter()
        losses_total = AverageMeter()
        losses_retrain = AverageMeter()

    BACKBONE.train()  # set to training mode

    model_without_ddp = BACKBONE.module if MULTI_GPU else BACKBONE

    regularization_terms = {}
    parms_without_ddp = {
        n: p for n, p in model_without_ddp.named_parameters() if p.requires_grad
    }

    # eval before training
    print("Perform Evaluation on forget train set and remain train set...")
    forget_acc_train_before = eval_data(
        BACKBONE, train_loader_forget, DEVICE, "forget-train", batch
    )
    remain_acc_train_before = eval_data(
        BACKBONE, train_loader_remain, DEVICE, "remain-train", batch
    )
    print("forget_acc_train_before", forget_acc_train_before)
    print("remain_acc_train_before", remain_acc_train_before)
    print("\n")
    print("Perform Evaluation on forget test set and remain test set...")
    forget_acc_before = eval_data(BACKBONE, testloader_forget, DEVICE, "forget", batch)
    remain_acc_before = eval_data(BACKBONE, testloader_remain, DEVICE, "remain", batch)
    open_acc_before = eval_data(BACKBONE, testloader_open, DEVICE, "open", batch)
    remain_all_acc_before = eval_data(
        BACKBONE, testloader_remain_all, DEVICE, "remain_all", batch
    )
    wandb.log(
        {
            "forget_acc_before": forget_acc_before,
            "remain_acc_before": remain_acc_before,
            "open_acc_before": open_acc_before,
            "remain_all_acc_before": remain_all_acc_before,
        }
    )
    if args.one_stage:
        BACKBONE.train()  # set to training mode
        print("start GS-LoRA training...")
        for epoch in range(NUM_EPOCH):  # start training process
            if args.warmup_alpha:
                if epoch < args.alpha_epoch:
                    args.alpha = 0
                else:
                    args.alpha = args.big_alpha
            if args.beta_decay:
                if epoch < 50:
                    args.beta = args.beta
                else:
                    args.beta = args.small_beta
            lr_scheduler.step(epoch)

            (
                batch,
                highest_H_mean,
                losses_forget,
                losses_remain,
                top1_forget,
                top1_remain,
                losses_total,
                losses_structure,
            ) = train_one_epoch(
                model=BACKBONE,
                dataloader_forget=train_loader_forget,
                dataloader_remain=train_loader_remain,
                testloader_forget=testloader_forget,
                testloader_remain=testloader_remain,
                dataloader_open=testloader_open,
                device=DEVICE,
                criterion=LOSS,
                optimizer=OPTIMIZER,
                epoch=epoch,
                batch=batch,
                losses_forget=losses_forget,
                top1_forget=top1_forget,
                losses_remain=losses_remain,
                top1_remain=top1_remain,
                losses_total=losses_total,
                losses_structure=losses_structure,
                beta=args.beta,
                BND=args.BND,
                forget_acc_before=forget_acc_before,
                highest_H_mean=highest_H_mean,
                cfg=cfg,
                alpha=args.alpha,
            )
        # print(batch)

    elif args.retrain:
        losses_total.reset()
        losses_CE.reset()
        losses_reg.reset()

        BACKBONE.train()  # set to training mode
        print("start retrain...")
        epoch = 0
        for epoch in range(NUM_EPOCH):  # start training process

            lr_scheduler.step(epoch)
            batch, highest_H_mean, losses_CE, losses_reg, losses_total = (
                train_one_epoch_regularzation(
                    model=BACKBONE,
                    criterion=LOSS,
                    data_loader_cl_forget=train_loader_remain,
                    optimizer=OPTIMIZER,
                    device=DEVICE,
                    epoch=epoch,
                    batch=batch,
                    reg_lambda=0,
                    regularization_terms=None,
                    losses_CE=losses_CE,
                    losses_regularization=losses_reg,
                    losses_total=losses_total,
                    task_i="0",
                    testloader_forget=testloader_forget,
                    testloader_remain=testloader_remain,
                    highest_H_mean=highest_H_mean,
                    forget_acc_before=forget_acc_before,
                    cfg=cfg,
                    testloader_open=testloader_open,
                )
            )

    else:  # CL baselines
        BACKBONE.train()
        print("start CL baselines...")
        # 1. backup the weight of current task
        task_param = {}
        model_without_ddp = BACKBONE.module if MULTI_GPU else BACKBONE
        for n, p in model_without_ddp.named_parameters():
            if p.requires_grad:
                task_param[n] = p.data.clone().detach()

        # 2. Calculate the Information Matrix
        if args.l2:

            def calculate_importance_l2(model, dataloader):
                # Use an identity importance so it is an L2 regularization.
                print("\033[32mcalculate_importance_l2\033[0m")
                importance = {}
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        importance[n] = p.clone().detach().fill_(1)  # Identity
                return importance

            importance = calculate_importance_l2(
                model_without_ddp, importance_dataloader_train
            )
            regularization_terms[0] = {
                "importance": importance,
                "task_param": task_param,
            }

        elif args.ewc:

            def calculate_importance_ewc(model, dataloader):
                print("\033[32mcalculate importance of ewc...\033[0m")

                # Initialize the importance matrix
                if args.online and len(regularization_terms) > 0:
                    importance = regularization_terms[0]["importance"]
                else:
                    importance = {}
                    for n, p in model_without_ddp.named_parameters():
                        if p.requires_grad:
                            importance[n] = (
                                p.clone().detach().fill_(0)
                            )  # zero initialization

                # Sample a subset (n_fisher_sample) of data to estimate the importance matrix (batch size is 1)
                # Otherwise it uses mini-batch to estimate the importance matrix. This speeds up the process a lot with similar performance.

                if args.n_fisher_sample is not None:
                    # FIXME: there is a bug RuntimeError:  Trying to resize storage that is not resizable
                    n_sample = min(args.n_fisher_sample, len(dataloader.dataset))
                    print(
                        "Sample",
                        args.n_fisher_sample,
                        "data to estimate the importance matrix",
                    )
                    rand_ind = random.sample(
                        list(range(len(dataloader.dataset))), n_sample
                    )
                    subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
                    subdataloader = DataLoader(
                        subdata,
                        batch_size=2,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                else:
                    subdataloader = dataloader

                model.eval()
                # Accumulate the square of gradients
                for i, (samples, targets) in enumerate(subdataloader):
                    samples = samples.to(DEVICE)

                    targets = targets.to(DEVICE)
                    # if dist.get_rank()==0:
                    #     import pdb; pdb.set_trace()
                    outputs, embeds = model(samples.float(), targets)

                    losses = LOSS(outputs, targets)

                    model.zero_grad()
                    losses.backward()
                    for n, p in importance.items():
                        if (
                            parms_without_ddp[n].grad is not None
                        ):  # some parameters may not have grad
                            p += (
                                (parms_without_ddp[n].grad ** 2)
                                * len(samples)
                                / len(subdataloader)
                            )

                model.train()
                return importance

            importance = calculate_importance_ewc(
                model_without_ddp, importance_dataloader_train
            )
            regularization_terms[0] = {
                "importance": importance,
                "task_param": task_param,
            }

        elif args.MAS:

            def calculate_importance_mas(model, dataloader):
                print("\033[32mcalculate importance of mas...\033[0m")

                # Initialize the importance matrix
                if args.online and len(regularization_terms) > 0:
                    importance = regularization_terms[0]["importance"]
                else:
                    importance = {}
                    for n, p in model_without_ddp.named_parameters():
                        if p.requires_grad:
                            importance[n] = (
                                p.clone().detach().fill_(0)
                            )  # zero initialization

                model.eval()
                # The square of the L2 norm of the network output logits is used as the loss,
                # and the partial derivative is taken to obtain the gradient.
                for i, (samples, targets) in enumerate(dataloader):
                    samples = samples.to(DEVICE)
                    targets = targets.to(DEVICE)

                    outputs, embeds = model(samples.float(), targets)

                    outputs.pow_(2)
                    loss = outputs.mean()

                    model.zero_grad()
                    loss.backward()

                    for n, p in importance.items():
                        if parms_without_ddp[n].grad is not None:
                            p += parms_without_ddp[n].grad.abs() / len(dataloader)

                model.train()
                return importance

            importance = calculate_importance_mas(
                model_without_ddp, importance_dataloader_train
            )
            regularization_terms[0] = {
                "importance": importance,
                "task_param": task_param,
            }

        # 1. learn current task
        epoch = 0
        if args.l2:
            reg_lambda = args.l2_lambda
        elif args.ewc:
            reg_lambda = args.ewc_lambda
        elif args.MAS:
            reg_lambda = args.mas_lambda

        for epoch in range(NUM_EPOCH):  # start training process
            lr_scheduler.step(epoch)
            batch, highest_H_mean, losses_CE, losses_reg, losses_total = (
                train_one_epoch_regularzation(
                    model=BACKBONE,
                    criterion=LOSS,
                    data_loader_cl_forget=train_loader_total,
                    optimizer=OPTIMIZER,
                    device=DEVICE,
                    epoch=epoch,
                    batch=batch,
                    reg_lambda=reg_lambda,
                    regularization_terms=regularization_terms,
                    losses_CE=losses_CE,
                    losses_regularization=losses_reg,
                    losses_total=losses_total,
                    task_i="0",
                    testloader_forget=testloader_forget,
                    testloader_remain=testloader_remain,
                    highest_H_mean=highest_H_mean,
                    forget_acc_before=forget_acc_before,
                    cfg=cfg,
                    testloader_open=testloader_open,
                )
            )

    remain_all_acc_after = eval_data(
        BACKBONE, testloader_remain_all, DEVICE, "remain_all", batch
    )
    print("remain_all_acc_after", remain_all_acc_after)
    wandb.log({"remain_all_acc_after": remain_all_acc_after})
    # calculate norm list
    if args.one_stage:
        norm_list = get_norm_of_lora(
            model_without_ddp,
            type="L2",
            group_num=args.vit_depth,
            group_type=args.grouping,
        )
        wandb.log({"norm_list": norm_list})
    # TODO:
    wandb.run.name = (
        "remain-"
        + str(args.num_of_first_cls)
        + "-forget-"
        + str(args.per_forget_cls)
        + "-lora_rank-"
        + str(args.lora_rank)
        + "beta"
        + str(args.beta)
        + "lr"
        + str(args.lr)
        + "BND"
        + str(args.BND)
        + "alpha"
        + str(args.alpha)
    )
    if args.warmup_alpha:
        wandb.run.name = wandb.run.name + "-warmup_alpha" + str(args.big_alpha)

    wandb.run.name = wandb.run.name + "-opencls" + str(args.open_cls_num)

    if args.ewc:
        wandb.run.name = "ewc_lambda" + str(args.ewc_lambda) + wandb.run.name
    elif args.MAS:
        wandb.run.name = "mas_lambda" + str(args.mas_lambda) + wandb.run.name
    elif args.l2:
        wandb.run.name = "l2_lambda" + str(args.l2_lambda) + wandb.run.name
    elif args.retrain:
        wandb.run.name = "retrain" + wandb.run.name
