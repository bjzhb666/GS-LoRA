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
    # lora pos (FFN and attention) on Transformer blocks
    parser.add_argument(
        "--lora_pos",
        type=str,
        default="FFN",
        help="lora pos (FFN and attention) on Transformer blocks (default: FFN)",
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
    print("=" * 60)

    wandb.login(key="808d6ef02f3a9c448c5641c132830eb0c3c83c2a")
    wandb.init(
        project="face recognition",
        group="casia100",
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
    dataset = datasets.ImageFolder(root=DATA_ROOT, transform=data_transform)

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_ROOT, "train"), transform=data_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_ROOT, "test"), transform=data_transform
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        drop_last=False,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        drop_last=False,
    )

    print("Number of Training Classes: {}".format(NUM_CLASS))

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
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        else:
            print(
                "No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(
                    BACKBONE_RESUME_ROOT
                )
            )
        print("=" * 60)

    if args.lora_rank > 0:
        lora.mark_only_lora_as_trainable(BACKBONE)
        print("Use LoRA in Transformer FFN, loar_rank: ", args.lora_rank)
    else:
        print("Do not use LoRA in Transformer FFN, train all parameters.")  # 68631040
    # count the number of learnable parameters
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
    DISP_FREQ = 10  # frequency to display training loss & acc
    VER_FREQ = 20  # frequency to perform validation

    batch = 0  # batch index

    losses = AverageMeter()
    top1 = AverageMeter()

    BACKBONE.train()  # set to training mode

    for epoch in range(NUM_EPOCH):  # start training process

        lr_scheduler.step(epoch)

        last_time = time.time()

        for inputs, labels in iter(trainloader):

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
                acc = []
                BACKBONE.eval()  # set to evaluation mode

                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in testloader:
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE).long()

                        outputs, _ = BACKBONE(images, labels)
                        # import pdb; pdb.set_trace()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                print("Test Accuracy: {:.2f}%".format(accuracy))
                wandb.log({"Test Accuracy": accuracy}, step=batch + 1)
                acc.append(accuracy)
                # save checkpoints per epoch

                if accuracy > highest_acc:
                    highest_acc = accuracy
                    if MULTI_GPU:
                        torch.save(
                            BACKBONE.module.state_dict(),
                            os.path.join(
                                WORK_PATH,
                                "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                    BACKBONE_NAME, epoch + 1, batch + 1, get_time()
                                ),
                            ),
                        )
                    else:
                        torch.save(
                            BACKBONE.state_dict(),
                            os.path.join(
                                WORK_PATH,
                                "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                    BACKBONE_NAME, epoch + 1, batch + 1, get_time()
                                ),
                            ),
                        )
                    # set the maximum checkpoint numbers to keep
                    if len(os.listdir(WORK_PATH)) >= 6:
                        checkpoints = list(
                            filter(lambda f: f.endswith(".pth"), os.listdir(WORK_PATH))
                        )
                        checkpoints.sort(
                            key=lambda f: os.path.getmtime(os.path.join(WORK_PATH, f))
                        )
                        os.remove(os.path.join(WORK_PATH, checkpoints[0]))
                BACKBONE.train()  # set to training mode

            batch += 1  # batch index

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
