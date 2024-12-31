import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb

from config import get_config
from image_iter import CLDatasetWrapper, CustomSubset

from util.utils import (
    get_val_data,
    perform_val,
    get_time,
    buffer_val,
    AverageMeter,
    train_accuracy,
    get_unique_classes,
    calculate_prototypes,
)
from util.utils import (
    split_dataset,
    count_trainable_parameters,
    reinitialize_lora_parameters,
    create_few_shot_dataset,
)
from util.args import get_args

import time
from vit_pytorch_face import ViT_face, ViT_face_low, ViT_face_up
from vit_pytorch_face import ViTs_face

# from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import loralib as lora
from engine_cl import train_one_epoch, eval_data, train_one_epoch_regularzation

# from torch.utils.data import Subset
from baselines.LIRFtrain import train_one_epoch_LIRF, eval_data_LIRF
from baselines.Lwftrain import train_one_epoch_Lwf
from baselines.DERtrain import train_one_epoch_DER
from baselines.FDRtrain import train_one_epoch_FDR
from baselines.SCRUBtrain import train_one_superepoch_SCRUB
from IPython import embed

import copy
from util.cal_norm import get_norm_of_lora
from torch.utils.data import DataLoader
import math


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)


if __name__ == "__main__":
    args = get_args()

    # ======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg["SEED"]  # random seed for reproduce results
    torch.manual_seed(SEED)

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
    elif args.data_mode == "tsne":
        NUM_CLASS = 10
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
    import random

    random.seed(SEED)
    random.shuffle(order_list)
    print("order_list", order_list)

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
            lora_pos=args.lora_pos,
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

    if args.one_stage:
        if args.lora_rank > 0:
            lora.mark_only_lora_as_trainable(BACKBONE)
            print("Use LoRA in Transformer FFN, loar_rank: ", args.lora_rank)
            # for n,p in BACKBONE.named_parameters():
            #     if 'loss.weight' in n: # open the gradient
            #         p.requires_grad = True
        else:
            print(
                "Do not use LoRA in Transformer FFN, train all parameters."
            )  # 19,157,504
    elif args.LIRF:

        teacher_model_low = ViT_face_low(
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
        )
        teacher_model_up = ViT_face_up(
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
        )
        student_model_low = ViT_face_low(
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
        )
        deposit_model_low = ViT_face_low(
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
        )

        print("Loading teacher model using the pretrained weights...")
        teacher_model_low.load_state_dict(
            torch.load(BACKBONE_RESUME_ROOT), strict=False
        )
        teacher_model_up.load_state_dict(torch.load(BACKBONE_RESUME_ROOT), strict=False)
        student_model_low.load_state_dict(
            torch.load(BACKBONE_RESUME_ROOT), strict=False
        )
        deposit_model_low.load_state_dict(
            torch.load(BACKBONE_RESUME_ROOT), strict=False
        )

        # freeze teacher model, teacher model and student model use the same upper part
        for n, p in teacher_model_low.named_parameters():
            p.requires_grad = False
        for n, p in teacher_model_up.named_parameters():
            p.requires_grad = False

        print("Teacher model up parameter names:")
        for n, p in teacher_model_up.named_parameters():
            print(n)
        print("\n")
        print("\n")

        print("Student model low parameter names:")
        for n, p in student_model_low.named_parameters():
            print(n)

        BACKBONE = student_model_low
        teacher_model_low = teacher_model_low.to(DEVICE)
        teacher_model_up = teacher_model_up.to(DEVICE)
        deposit_model_low = deposit_model_low.to(DEVICE)

        deposit_model_low.train()

    else:  # CL baselines(EWC, MAS, L2, Lwf, DER, DER++, FDR) and SCRUB
        for n, p in BACKBONE.named_parameters():
            if "loss" in n and not args.ffn_open:  # freeze the last layer
                p.requires_grad = False

        if args.only_ffn:
            for n, p in BACKBONE.named_parameters():
                if "fn.fn.net" in n:
                    p.requires_grad = True
                elif "loss" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        if args.SCRUB:  # deep copy to create two independent models
            teacher_model = copy.deepcopy(BACKBONE)
            teacher_model = teacher_model.to(DEVICE)
            teacher_model.eval()

            beta = 0.1

            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return (1 - beta) * averaged_model_parameter + beta * model_parameter

            swa_model = torch.optim.swa_utils.AveragedModel(BACKBONE, avg_fn=avg_fn)
            swa_model = swa_model.to(DEVICE)

        if args.Lwf:
            teacher_model = copy.deepcopy(BACKBONE)
            teacher_model = teacher_model.to(DEVICE)
            teacher_model.eval()
            # freeze teacher model
            for n, p in teacher_model.named_parameters():
                p.requires_grad = False

        if args.Der:
            teacher_model = copy.deepcopy(BACKBONE)
            teacher_model = teacher_model.to(DEVICE)
            teacher_model.eval()
            # freeze teacher model
            for n, p in teacher_model.named_parameters():
                p.requires_grad = False

        if args.FDR:
            teacher_model = copy.deepcopy(BACKBONE)
            teacher_model = teacher_model.to(DEVICE)
            teacher_model.eval()
            # freeze teacher model
            for n, p in teacher_model.named_parameters():
                p.requires_grad = False

    # get the number of learnable parameters
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

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    BACKBONE.train()  # set to training mode

    model_without_ddp = BACKBONE.module if MULTI_GPU else BACKBONE

    regularization_terms = {}  # store the regularization terms

    for task_i in range(args.num_tasks):
        print("\n")
        print(
            "\033[34m=========================task:{}==============================\033[0m".format(
                task_i
            )
        )  # blue
        print("\n")
        # load pretrained model when task_i > 0
        if task_i > 0 and args.one_stage:
            print("load pretrained model in task {}".format(task_i - 1))
            BACKBONE.load_state_dict(
                torch.load(
                    os.path.join(
                        WORK_PATH,
                        "task-level",
                        "Backbone_task_{}.pth".format(task_i - 1),
                    )
                )
            )
            # reinitialize LoRA model
            reinitialize_lora_parameters(model_without_ddp)
        # split datasets
        # 1. calculate st1, en1, st2, en2
        st1 = args.open_cls_num
        en1 = args.num_of_first_cls - task_i * args.per_forget_cls
        st2 = en1
        en2 = en1 + args.per_forget_cls
        if task_i > 0:  # not the first task
            old_st = en2
            old_en = NUM_CLASS
        # 2. split datasets
        train_dataset = datasets.ImageFolder(
            root=os.path.join(DATA_ROOT, "train"), transform=data_transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(DATA_ROOT, "test"), transform=data_transform
        )
        _, forget_dataset_train = split_dataset(
            dataset=train_dataset,
            class_order_list=order_list,
            split1_start=st1,
            split1_end=en1,
            split2_start=st2,
            split2_end=en2,
        )
        open_dataset_train, remain_dataset_train = split_dataset(
            dataset=train_dataset,
            class_order_list=order_list,
            split1_start=0,
            split1_end=st1,
            split2_start=st1,
            split2_end=en1,
        )
        _, forget_dataset_test = split_dataset(
            dataset=test_dataset,
            class_order_list=order_list,
            split1_start=st1,
            split1_end=en1,
            split2_start=st2,
            split2_end=en2,
        )
        open_dataset_test, remain_dataset_test = split_dataset(
            dataset=test_dataset,
            class_order_list=order_list,
            split1_start=0,
            split1_end=st1,
            split2_start=st1,
            split2_end=en1,
        )
        # get remain all datasets
        remain_dataset_test_all = torch.utils.data.ConcatDataset(
            [remain_dataset_test, open_dataset_test]
        )
        # import pdb; pdb.set_trace()
        # Output all categories of remain dataset train
        num_remain_classes, num_remain_total = get_unique_classes(
            remain_dataset_train, train_dataset
        )
        num_forget_classes, num_forget_total = get_unique_classes(
            forget_dataset_train, train_dataset
        )
        num_open_classes, num_open_total = get_unique_classes(
            open_dataset_train, train_dataset
        )
        print("remain", num_remain_classes, len(num_remain_classes))
        print("forget", num_forget_classes, len(num_forget_classes))
        print("open", num_open_classes, len(num_open_classes))
        # import pdb; pdb.set_trace()
        print(
            "\033[31m" + "Same class?" + "\033[0m",
            bool(set(num_remain_classes) & set(num_forget_classes)),
        )
        print(
            "\033[31m" + "Same class?" + "\033[0m",
            bool(set(num_remain_classes) & set(num_open_classes)),
        )
        print(
            "\033[31m" + "Same class?" + "\033[0m",
            bool(set(num_open_classes) & set(num_forget_classes)),
        )
        # get sub datasets
        len_forget_dataset_train = len(forget_dataset_train)
        len_remain_dataset_train = len(remain_dataset_train)

        if args.few_shot:
            subset_size_forget = args.few_shot_num
            subset_size_remain = args.few_shot_num
            forget_dataset_train_sub = create_few_shot_dataset(
                forget_dataset_train, subset_size_forget
            )
            remain_dataset_train_sub = create_few_shot_dataset(
                remain_dataset_train, subset_size_remain
            )

            # debug code for few shot

            print(f"Original dataset size: {len(forget_dataset_train)}")
            print(f"Few-shot dataset size: {len(forget_dataset_train_sub)}")

            from collections import Counter

            few_shot_labels = [
                forget_dataset_train.targets[idx]
                for idx in forget_dataset_train_sub.indices
            ]
            label_counts = Counter(few_shot_labels)
            print("number of samples per class:", label_counts)

        else:
            subset_size_forget = int(len_forget_dataset_train * args.data_ratio)
            subset_size_remain = int(len_remain_dataset_train * args.data_ratio)

            subset_indices_forget = torch.randperm(len_forget_dataset_train)[
                :subset_size_forget
            ]
            subset_indices_remain = torch.randperm(len_remain_dataset_train)[
                :subset_size_remain
            ]

            forget_dataset_train_sub = CustomSubset(
                forget_dataset_train, subset_indices_forget
            )
            remain_dataset_train_sub = CustomSubset(
                remain_dataset_train, subset_indices_remain
            )
        if args.prototype:
            print("Calculate prototype...")
            # calculate the prototype for all classes
            total_dataset_train = torch.utils.data.ConcatDataset(
                [forget_dataset_train_sub, remain_dataset_train_sub]
            )
            prototype = calculate_prototypes(
                backbone=BACKBONE,
                dataset=total_dataset_train,
                device=DEVICE,
                batch_size=2000,
            )
        else:
            prototype = None

        if task_i == 0:
            if args.few_shot:
                importance_dataset_train = create_few_shot_dataset(
                    remain_dataset_train, args.few_shot_num
                )
            else:
                # create importance dataset and dataloader
                len_importance_dataset_train = len(remain_dataset_train)
                subset_size_importance = int(
                    len_importance_dataset_train * args.data_ratio
                )
                subset_indices_importance = torch.randperm(
                    len_importance_dataset_train
                )[:subset_size_importance]
                importance_dataset_train = CustomSubset(
                    remain_dataset_train, subset_indices_importance
                )

            name_importance_classes, num_importance_classes = get_unique_classes(
                importance_dataset_train, train_dataset
            )
            print(
                "importance class for task 0: ",
                name_importance_classes,
                num_importance_classes,
            )

            importance_generator = torch.Generator()
            importance_generator.manual_seed(SEED)

            importance_dataloader_train = torch.utils.data.DataLoader(
                importance_dataset_train,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=WORKERS,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=importance_generator,
            )

        # prepare datset for regularzation method
        if not args.one_stage and not args.SCRUB:  # CL baselines
            forget_dataset_train_sub = CLDatasetWrapper(forget_dataset_train_sub)
            if args.replay:
                print("\033[33mConcat datsets\033[0m")
                total_dataset_train = torch.utils.data.ConcatDataset(
                    [forget_dataset_train_sub, remain_dataset_train_sub]
                )
        forget_train_generator = torch.Generator()
        forget_train_generator.manual_seed(SEED)
        remain_train_generator = torch.Generator()
        remain_train_generator.manual_seed(SEED)

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

        print("len(train_loader_forget)", len(train_loader_forget))
        print("len(train_loader_remain)", len(train_loader_remain))
        print("len(testloader_forget)", len(testloader_forget))
        print("len(testloader_remain)", len(testloader_remain))
        print("len(testloader_open)", len(testloader_open))

        if task_i > 0:
            _, old_dataset_test = split_dataset(
                dataset=test_dataset,
                class_order_list=order_list,
                split1_start=0,
                split1_end=old_st,
                split2_start=old_st,
                split2_end=old_en,
            )

            name_old_classes, num_old_classes = get_unique_classes(
                old_dataset_test, test_dataset
            )
            print("old class", name_old_classes, num_old_classes)

            testloader_old = torch.utils.data.DataLoader(
                old_dataset_test,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=WORKERS,
                drop_last=False,
            )
            print("\n")
            print("len(testloader_old)", len(testloader_old))

        if args.replay:
            total_train_generator = torch.Generator()
            total_train_generator.manual_seed(SEED)
            train_loader_total = torch.utils.data.DataLoader(
                total_dataset_train,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=WORKERS,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=total_train_generator,
            )
        # import pdb; pdb.set_trace()
        # testloader = torch.utils.data.DataLoader(test_dataset,
        #                                          batch_size=BATCH_SIZE,
        #                                          shuffle=False,
        #                                          num_workers=WORKERS,
        #                                          drop_last=False)

        # print("Number of Training Classes: {}".format(NUM_CLASS))

        highest_H_mean = 0.0

        # embed()
        # ======= model & loss & optimizer =======#
        if args.epochs != 0:
            # LOSS = LossFaceCE(type=HEAD_NAME,dim=512,num_class=NUM_CLASS, GPU_ID=GPU_ID)
            LOSS = nn.CrossEntropyLoss()
            # embed()
            OPTIMIZER = create_optimizer(
                args, BACKBONE
            )  # create again to reinitialize optimizer
            print("=" * 60)
            print(OPTIMIZER)
            print("Optimizer Generated")
            print("=" * 60)
            lr_scheduler, _ = create_scheduler(
                args, OPTIMIZER
            )  # create again to reinitialize lr_scheduler

        # ======= train & validation & save checkpoint =======#

        batch = 0  # batch index

        if args.one_stage:
            losses_forget = AverageMeter()
            top1_forget = AverageMeter()
            losses_remain = AverageMeter()
            top1_remain = AverageMeter()
            losses_total = AverageMeter()
            losses_structure = AverageMeter()
            losses_prototype_forget = AverageMeter()
            losses_prototype_remain = AverageMeter()
        elif args.LIRF:
            # losses_CE, losses_AT, kd_lossesKP, losses_pt_re, losses_total, losses_remain
            losses_CE = AverageMeter()
            losses_AT = AverageMeter()
            kd_lossesKP = AverageMeter()
            losses_pt_re = AverageMeter()
            losses_total = AverageMeter()
            losses_remain = AverageMeter()
        elif args.SCRUB:
            losses_CE = AverageMeter()
            losses_kd_remain = AverageMeter()
            losses_kd_forget = AverageMeter()
            losses_total_forget = AverageMeter()
            losses_total_remain = AverageMeter()

            # change the optimizer in SCRUB
            trainable_list = nn.ModuleList([])
            trainable_list.append(BACKBONE)

            if args.opt == "sgd":
                OPTIMIZER = optim.SGD(
                    trainable_list.parameters(),
                    lr=args.sgda_learning_rate,
                    momentum=args.sgda_momentum,
                    weight_decay=args.sgda_weight_decay,
                )
            elif args.opt == "adam":
                OPTIMIZER = optim.Adam(
                    trainable_list.parameters(),
                    lr=args.sgda_learning_rate,
                    weight_decay=args.sgda_weight_decay,
                )
            elif args.opt == "rmsp":
                OPTIMIZER = optim.RMSprop(
                    trainable_list.parameters(),
                    lr=args.sgda_learning_rate,
                    momentum=args.sgda_momentum,
                    weight_decay=args.sgda_weight_decay,
                )
        elif args.Lwf:
            losses_CE = AverageMeter()
            losses_kd = AverageMeter()
            losses_total = AverageMeter()
            losses_remain = AverageMeter()
        elif args.Der:
            losses_CE = AverageMeter()
            losses_der = AverageMeter()
            losses_total = AverageMeter()
        elif args.FDR:
            losses_CE = AverageMeter()
            losses_FDR = AverageMeter()
            losses_total = AverageMeter()
        else:  # CL baselines
            losses_CE = AverageMeter()
            losses_reg = AverageMeter()
            losses_total = AverageMeter()
            losses_retrain = AverageMeter()

        if not args.LIRF:
            # eval before training
            print("Perform Evaluation on forget train set and remain train set...")
            forget_acc_train_before = eval_data(
                BACKBONE,
                train_loader_forget,
                DEVICE,
                "forget-train-{}".format(task_i),
                batch,
            )
            remain_acc_train_before = eval_data(
                BACKBONE,
                train_loader_remain,
                DEVICE,
                "remain-train-{}".format(task_i),
                batch,
            )
            print("forget_acc_train_before-{}".format(task_i), forget_acc_train_before)
            print("remain_acc_train_before-{}".format(task_i), remain_acc_train_before)
            print("\n")
            print("Perform Evaluation on forget test set and remain test set...")
            forget_acc_before = eval_data(
                BACKBONE,
                testloader_forget,
                DEVICE,
                "forget-before-{}".format(task_i),
                batch,
            )
            remain_acc_before = eval_data(
                BACKBONE,
                testloader_remain,
                DEVICE,
                "remain-before-{}".format(task_i),
                batch,
            )
            open_acc_before = eval_data(
                BACKBONE,
                testloader_open,
                DEVICE,
                "open-before-{}".format(task_i),
                batch,
            )
            remain_all_acc_before = eval_data(
                BACKBONE,
                testloader_remain_all,
                DEVICE,
                "remain-all-before-{}".format(task_i),
                batch,
            )

            wandb.log(
                {
                    "forget_acc_before_{}".format(task_i): forget_acc_before,
                    "remain_acc_before_{}".format(task_i): remain_acc_before,
                    "open_acc_before_{}".format(task_i): open_acc_before,
                    "remain_all_acc_before_{}".format(task_i): remain_all_acc_before,
                }
            )
            if task_i > 0:
                # eval old test set
                old_acc_before = eval_data(
                    BACKBONE, testloader_old, DEVICE, "old-{}".format(task_i), batch
                )
                wandb.log({"old_acc_before_{}".format(task_i): old_acc_before})
        else:
            # eval before training LIRF
            print("LIRF Perform Evaluation on forget train set and remain train set...")
            forget_acc_train_before = eval_data_LIRF(
                student_low=BACKBONE,
                teacher_up=teacher_model_up,
                testloader=train_loader_forget,
                device=DEVICE,
                mode="forget-train-{}".format(task_i),
                batch=batch,
            )
            remain_acc_train_before = eval_data_LIRF(
                student_low=BACKBONE,
                teacher_up=teacher_model_up,
                testloader=train_loader_remain,
                device=DEVICE,
                mode="remain-train-{}".format(task_i),
                batch=batch,
            )
            print("forget_acc_train_before-{}".format(task_i), forget_acc_train_before)
            print("remain_acc_train_before-{}".format(task_i), remain_acc_train_before)
            print("\n")
            print("LIRF Perform Evaluation on forget test set and remain test set...")
            forget_acc_before = eval_data_LIRF(
                student_low=BACKBONE,
                teacher_up=teacher_model_up,
                testloader=testloader_forget,
                device=DEVICE,
                mode="forget-{}".format(task_i),
                batch=batch,
            )
            remain_acc_before = eval_data_LIRF(
                student_low=BACKBONE,
                teacher_up=teacher_model_up,
                testloader=testloader_remain,
                device=DEVICE,
                mode="remain-{}".format(task_i),
                batch=batch,
            )
            open_acc_before = eval_data_LIRF(
                student_low=BACKBONE,
                teacher_up=teacher_model_up,
                testloader=testloader_open,
                device=DEVICE,
                mode="open-{}".format(task_i),
                batch=batch,
            )
            remain_all_acc_before = eval_data_LIRF(
                student_low=BACKBONE,
                teacher_up=teacher_model_up,
                testloader=testloader_remain_all,
                device=DEVICE,
                mode="remain-all-{}".format(task_i),
                batch=batch,
            )
            wandb.log(
                {
                    "forget_acc_before_{}".format(task_i): forget_acc_before,
                    "remain_acc_before_{}".format(task_i): remain_acc_before,
                    "open_acc_before_{}".format(task_i): open_acc_before,
                    "remain_all_acc_before_{}".format(task_i): remain_all_acc_before,
                }
            )
            if task_i > 0:
                # eval old test set
                old_acc_before = eval_data_LIRF(
                    student_low=BACKBONE,
                    teacher_up=teacher_model_up,
                    testloader=testloader_old,
                    device=DEVICE,
                    mode="old-{}".format(task_i),
                    batch=batch,
                )
                wandb.log({"old_acc_before_{}".format(task_i): old_acc_before})

        parms_without_ddp = {
            n: p for n, p in model_without_ddp.named_parameters() if p.requires_grad
        }  # for convenience

        if args.one_stage:
            cl_beta = args.cl_beta_list[task_i]
            BACKBONE.train()  # set to training mode
            print("start one stage forget remain training...")
            epoch = 0  # force it to be 0 to avoid affecting the epoch calculation of the next task
            for epoch in range(NUM_EPOCH):  # start training process
                if args.warmup_alpha:
                    if epoch < args.alpha_epoch:
                        args.alpha = 0
                    else:
                        args.alpha = args.big_alpha

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
                    losses_prototype_forget,
                    losses_prototype_remain,
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
                    beta=cl_beta,
                    BND=args.BND,
                    forget_acc_before=forget_acc_before,
                    highest_H_mean=highest_H_mean,
                    cfg=cfg,
                    alpha=args.alpha,
                    task_i=task_i,
                    use_prototype=args.prototype,
                    prototype_dict=prototype,
                    prototype_weight_forget=args.pro_f_weight,
                    prototype_weight_remain=args.pro_r_weight,
                    losses_prototype_forget=losses_prototype_forget,
                    losses_prototype_remain=losses_prototype_remain,
                )
                # print(batch)
                # calculate norm list

            norm_list = get_norm_of_lora(
                model_without_ddp, type="L2", group_num=args.vit_depth
            )
            wandb.log({"norm_list-{}".format(task_i): norm_list})

        elif args.retrain:
            losses_total.reset()
            losses_CE.reset()
            losses_reg.reset()

            BACKBONE.train()
            print("start retrain...")
            # reinitalize the model
            BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
            epoch = 0  # force it to be 0 to avoid affecting the epoch calculation of the next task
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
                        task_i=task_i,
                        testloader_forget=testloader_forget,
                        testloader_remain=testloader_remain,
                        highest_H_mean=highest_H_mean,
                        forget_acc_before=forget_acc_before,
                        cfg=cfg,
                        testloader_open=testloader_open,
                    )
                )

        elif args.LIRF:
            BACKBONE.train()
            deposit_model_low.train()

            print("start LIRF training...")
            epoch = 0  # force it to be 0 to avoid affecting the epoch calculation of the next task

            for epoch in range(NUM_EPOCH):  # start training process
                lr_scheduler.step(epoch)

                (
                    batch,
                    highest_H_mean,
                    losses_CE,
                    losses_AT,
                    kd_lossesKP,
                    losses_pt_re,
                    losses_total,
                    losses_remain,
                ) = train_one_epoch_LIRF(
                    student_low=BACKBONE,
                    deposit_low=deposit_model_low,
                    teacher_low=teacher_model_low,
                    teacher_up=teacher_model_up,
                    data_loader_cl_forget=train_loader_forget,
                    remain_loader=train_loader_remain,
                    criterion=LOSS,
                    optimizer=OPTIMIZER,
                    device=DEVICE,
                    epoch=epoch,
                    batch=batch,
                    losses_CE=losses_CE,
                    losses_AT=losses_AT,
                    kd_lossesKP=kd_lossesKP,
                    losses_pt_re=losses_pt_re,
                    losses_total=losses_total,
                    losses_remain=losses_remain,
                    task_i=task_i,
                    testloader_forget=testloader_forget,
                    testloader_remain=testloader_remain,
                    highest_H_mean=highest_H_mean,
                    forget_acc_before=forget_acc_before,
                    cfg=cfg,
                    testloader_open=testloader_open,
                )

        elif args.SCRUB:
            losses_CE.reset()
            losses_kd_remain.reset()
            losses_kd_forget.reset()
            losses_total_forget.reset()
            losses_total_remain.reset()

            BACKBONE.train()
            teacher_model.eval()

            print("start SCRUB training...")
            epoch = 0
            for epoch in range(args.SCRUB_superepoch):
                # lr_scheduler.step(epoch)
                (
                    batch,
                    highest_H_mean,
                    losses_CE,
                    losses_total_forget,
                    losses_total_remain,
                    losses_kd_forget,
                    losses_kd_remain,
                    swa_model,
                ) = train_one_superepoch_SCRUB(
                    student=BACKBONE,
                    teacher=teacher_model,
                    swa_model=swa_model,
                    data_loader_forget=train_loader_forget,
                    remain_loader=train_loader_remain,
                    criterion=LOSS,
                    optimizer=OPTIMIZER,
                    device=DEVICE,
                    superepoch=epoch,
                    batch=batch,
                    losses_CE=losses_CE,
                    losses_total_forget=losses_total_forget,
                    losses_total_remain=losses_total_remain,
                    losses_kd_forget=losses_kd_forget,
                    losses_kd_remain=losses_kd_remain,
                    task_i=task_i,
                    testloader_forget=testloader_forget,
                    testloader_remain=testloader_remain,
                    highest_H_mean=highest_H_mean,
                    forget_acc_before=forget_acc_before,
                    cfg=cfg,
                    kd_T=args.kd_T,
                    sgda_smoothing=args.sgda_smoothing,
                    sgda_gamma=args.sgda_gamma,
                    sgda_alpha=args.sgda_alpha,
                    testloader_open=testloader_open,
                )

        elif args.Lwf:
            losses_CE.reset()
            losses_kd.reset()
            losses_total.reset()
            losses_remain.reset()

            BACKBONE.train()
            teacher_model.eval()

            print("start Lwf training...")
            epoch = 0
            for epoch in range(NUM_EPOCH):
                lr_scheduler.step(epoch)
                (
                    batch,
                    highest_H_mean,
                    losses_CE,
                    losses_kd,
                    losses_remain,
                    losses_total,
                ) = train_one_epoch_Lwf(
                    student_model=BACKBONE,
                    teacher_model=teacher_model,
                    data_loader_cl_forget=train_loader_forget,
                    remain_loader=train_loader_remain,
                    criterion=LOSS,
                    optimizer=OPTIMIZER,
                    device=DEVICE,
                    epoch=epoch,
                    batch=batch,
                    losses_CE=losses_CE,
                    losses_total=losses_total,
                    losses_KD=losses_kd,
                    losses_remain=losses_remain,
                    task_i=task_i,
                    testloader_forget=testloader_forget,
                    testloader_remain=testloader_remain,
                    highest_H_mean=highest_H_mean,
                    forget_acc_before=forget_acc_before,
                    cfg=cfg,
                    lambda_kd=args.Lwf_lambda_kd,
                    lambda_remain=args.Lwf_lambda_remain,
                    temperature=args.Lwf_T,
                    testloader_open=testloader_open,
                )

        elif args.Der:
            losses_CE.reset()
            losses_der.reset()
            losses_total.reset()

            BACKBONE.train()
            teacher_model.eval()

            print("start DER training...")
            print("DER ++ is ", args.DER_plus)
            epoch = 0

            for epoch in range(NUM_EPOCH):
                lr_scheduler.step(epoch)
                batch, highest_H_mean, losses_CE, losses_total, losses_der = (
                    train_one_epoch_DER(
                        student_model=BACKBONE,
                        teacher_model=teacher_model,
                        data_loader_cl_forget=train_loader_forget,
                        remain_loader=train_loader_remain,
                        criterion=LOSS,
                        optimizer=OPTIMIZER,
                        device=DEVICE,
                        epoch=epoch,
                        batch=batch,
                        losses_CE=losses_CE,
                        losses_total=losses_total,
                        losses_DER=losses_der,
                        task_i=task_i,
                        testloader_forget=testloader_forget,
                        testloader_remain=testloader_remain,
                        highest_H_mean=highest_H_mean,
                        forget_acc_before=forget_acc_before,
                        cfg=cfg,
                        lambda_der=args.DER_lambda,
                        plus=args.DER_plus,
                        lambda_der_plus=args.DER_plus_lambda,
                        testloader_open=testloader_open,
                    )
                )

        elif args.FDR:
            losses_CE.reset()
            losses_FDR.reset()
            losses_total.reset()

            BACKBONE.train()
            teacher_model.eval()

            print("start FDR training...")
            epoch = 0
            for epoch in range(NUM_EPOCH):
                lr_scheduler.step(epoch)
                batch, highest_H_mean, losses_CE, losses_FDR, losses_total = (
                    train_one_epoch_FDR(
                        student_model=BACKBONE,
                        teacher_model=teacher_model,
                        data_loader_cl_forget=train_loader_forget,
                        remain_loader=train_loader_remain,
                        criterion=LOSS,
                        optimizer=OPTIMIZER,
                        device=DEVICE,
                        epoch=epoch,
                        batch=batch,
                        losses_CE=losses_CE,
                        losses_total=losses_total,
                        losses_FDR=losses_FDR,
                        task_i=task_i,
                        testloader_forget=testloader_forget,
                        testloader_remain=testloader_remain,
                        highest_H_mean=highest_H_mean,
                        forget_acc_before=forget_acc_before,
                        cfg=cfg,
                        reg_lambda=args.FDR_lambda,
                        testloader_open=testloader_open,
                    )
                )

        else:  # CL baselines
            BACKBONE.train()

            if task_i == 0:
                # 1. Backup the weight of current task
                task_param = {}
                model_without_ddp = BACKBONE.module if MULTI_GPU else BACKBONE
                for n, p in model_without_ddp.named_parameters():
                    if p.requires_grad:
                        task_param[n] = p.clone().detach()

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

                        if args.n_fisher_sample is not None:  # We use None here
                            # FIXME: there is a bug RuntimeError:  Trying to resize storage that is not resizable
                            n_sample = min(
                                args.n_fisher_sample, len(dataloader.dataset)
                            )
                            print(
                                "Sample",
                                args.n_fisher_sample,
                                "data to estimate the importance matrix",
                            )
                            rand_ind = random.sample(
                                list(range(len(dataloader.dataset))), n_sample
                            )
                            subdata = torch.utils.data.Subset(
                                dataloader.dataset, rand_ind
                            )
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
                        # and the gradient is obtained by taking a partial derivative of it
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
                                    p += parms_without_ddp[n].grad.abs() / len(
                                        dataloader
                                    )

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

            for epoch in range(NUM_EPOCH):
                lr_scheduler.step(epoch)
                if args.replay:  # use train_loader_total
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
                            task_i=task_i,
                            testloader_forget=testloader_forget,
                            testloader_remain=testloader_remain,
                            highest_H_mean=highest_H_mean,
                            forget_acc_before=forget_acc_before,
                            cfg=cfg,
                            testloader_open=testloader_open,
                        )
                    )
                else:
                    batch, highest_H_mean, losses_CE, losses_reg, losses_total = (
                        train_one_epoch_regularzation(
                            model=BACKBONE,
                            criterion=LOSS,
                            data_loader_cl_forget=train_loader_forget,
                            optimizer=OPTIMIZER,
                            device=DEVICE,
                            epoch=epoch,
                            batch=batch,
                            reg_lambda=reg_lambda,
                            regularization_terms=regularization_terms,
                            losses_CE=losses_CE,
                            losses_regularization=losses_reg,
                            losses_total=losses_total,
                            task_i=task_i,
                            testloader_forget=testloader_forget,
                            testloader_remain=testloader_remain,
                            highest_H_mean=highest_H_mean,
                            forget_acc_before=forget_acc_before,
                            cfg=cfg,
                            testloader_open=testloader_open,
                        )
                    )
            # 2. Backup the weight of current task
            task_param = {}
            model_without_ddp = BACKBONE.module if MULTI_GPU else BACKBONE
            for n, p in model_without_ddp.named_parameters():
                if p.requires_grad:
                    task_param[n] = p.clone().detach()

            # 3. Calculate the Information Matrix and get the regularization terms
            # calculate the importance for the next task
            if task_i < args.num_tasks - 1:
                import_st1 = args.open_cls_num
                import_en1 = (
                    args.num_of_first_cls - (task_i + 1) * args.per_forget_cls
                )  # task_i + 1 is the next task
                import_st2 = import_en1
                import_en2 = import_en1 + args.per_forget_cls
                importance_dataset_train, _ = (
                    split_dataset(  # remain set, forget set, we just use the remain set of the next task
                        dataset=train_dataset,
                        class_order_list=order_list,
                        split1_start=import_st1,
                        split1_end=import_en1,
                        split2_start=import_st2,
                        split2_end=import_en2,
                    )
                )
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
                print(
                    "importance class", name_importance_classes, num_importance_classes
                )
                # import pdb; pdb.set_trace()
                if args.l2:
                    importance = calculate_importance_l2(
                        model_without_ddp, dataloader=importance_dataloader_train
                    )
                elif args.ewc:
                    importance = calculate_importance_ewc(
                        model_without_ddp, dataloader=importance_dataloader_train
                    )
                elif args.MAS:
                    importance = calculate_importance_mas(
                        model_without_ddp, dataloader=importance_dataloader_train
                    )
                if args.online and len(regularization_terms) > 0:
                    regularization_terms[0] = {
                        "importance": importance,
                        "task_param": task_param,
                    }
                else:
                    regularization_terms[task_i + 1] = {
                        "importance": importance,
                        "task_param": task_param,
                    }
            else:  # The last task, no need to calculate the importance
                print("The last task, no need to calculate the importance")

        # test for old classes after training task_i
        # save the model after one task training
        if args.one_stage:
            BACKBONE.eval()
            os.makedirs(os.path.join(WORK_PATH, "task-level"), exist_ok=True)
            torch.save(
                BACKBONE.state_dict(),
                os.path.join(
                    WORK_PATH, "task-level", "Backbone_task_{}.pth".format(task_i)
                ),
            )
            BACKBONE.train()
        else:
            BACKBONE.eval()
            os.makedirs(os.path.join(WORK_PATH, "task-level"), exist_ok=True)
            torch.save(
                BACKBONE.state_dict(),
                os.path.join(
                    WORK_PATH, "task-level", "Backbone_task_{}.pth".format(task_i)
                ),
            )
            if args.LIRF:
                deposit_model_low.eval()
                torch.save(
                    deposit_model_low.state_dict(),
                    os.path.join(
                        WORK_PATH,
                        "task-level",
                        "deposit_model_low_task_{}.pth".format(task_i),
                    ),
                )
                deposit_model_low.train()

                teacher_model_up.eval()
                torch.save(
                    teacher_model_up.state_dict(),
                    os.path.join(
                        WORK_PATH,
                        "task-level",
                        "teacher_model_up_task_{}.pth".format(task_i),
                    ),
                )

            BACKBONE.train()
        if task_i > 0:
            if not args.LIRF:
                old_acc = eval_data(
                    BACKBONE, testloader_old, DEVICE, "old-{}".format(task_i), batch
                )
            else:
                old_acc = eval_data_LIRF(
                    student_low=BACKBONE,
                    teacher_up=teacher_model_up,
                    testloader=testloader_old,
                    device=DEVICE,
                    mode="old-{}".format(task_i),
                    batch=batch,
                )

            wandb.log({"old_acc_after_{}".format(task_i): old_acc})
        # test remain all acc after learning one task
        if not args.LIRF:
            remain_all_acc_after = eval_data(
                BACKBONE,
                testloader_remain_all,
                DEVICE,
                "remain_all-{}".format(task_i),
                batch,
            )
        else:
            remain_all_acc_after = eval_data_LIRF(
                student_low=BACKBONE,
                teacher_up=teacher_model_up,
                testloader=testloader_remain_all,
                device=DEVICE,
                mode="remain_all-{}".format(task_i),
                batch=batch,
            )
        wandb.log({"remain_all_acc_after_{}".format(task_i): remain_all_acc_after})

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
    )
    if args.ewc:
        wandb.run.name = "ewc" + str(args.ewc_lambda) + wandb.run.name
    elif args.MAS:
        wandb.run.name = "mas" + str(args.mas_lambda) + wandb.run.name
    elif args.l2:
        wandb.run.name = "l2" + str(args.l2_lambda) + wandb.run.name
    elif args.retrain:
        wandb.run.name = "retrain-" + wandb.run.name
    elif args.LIRF:
        wandb.run.name = "LIRF" + wandb.run.name
    elif args.SCRUB:
        wandb.run.name = "SCRUB" + str(args.sgda_smoothing) + wandb.run.name
    elif args.Lwf:
        wandb.run.name = "Lwf" + wandb.run.name
    elif args.Der:
        wandb.run.name = (
            "DER" + str(args.DER_plus) + str(args.DER_lambda) + wandb.run.name
        )
    elif args.FDR:
        wandb.run.name = "FDR" + str(args.FDR_lambda) + wandb.run.name
    if args.few_shot:
        wandb.run.name = (
            "few_shot-"
            + str(args.few_shot_num)
            + "epoch-"
            + str(NUM_EPOCH)
            + wandb.run.name
        )
    wandb.run.name = "open" + str(args.open_cls_num) + wandb.run.name
