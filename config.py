import torch, os
import yaml
from IPython import embed


def get_config(args):
    configuration = dict(
        SEED=1337,  # random seed for reproduce results
        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        EMBEDDING_SIZE=512,  # feature dimension
    )

    if args.workers_id == "cpu" or not torch.cuda.is_available():
        configuration["GPU_ID"] = []
        print("check", args.workers_id, torch.cuda.is_available())
    else:
        configuration["GPU_ID"] = [int(i) for i in args.workers_id.split(",")]
    if len(configuration["GPU_ID"]) == 0:
        configuration["DEVICE"] = torch.device("cpu")
        configuration["MULTI_GPU"] = False
    else:
        configuration["DEVICE"] = torch.device("cuda:%d" % configuration["GPU_ID"][0])
        if len(configuration["GPU_ID"]) == 1:
            configuration["MULTI_GPU"] = False
        else:
            configuration["MULTI_GPU"] = True

    configuration["NUM_EPOCH"] = args.epochs
    configuration["BATCH_SIZE"] = args.batch_size
    configuration["WORKERS"] = args.num_workers

    if args.data_mode == "retina":
        configuration["DATA_ROOT"] = "./Data/ms1m-retinaface-t1/"
    elif args.data_mode == "casia":
        configuration["DATA_ROOT"] = "./data/faces_webface_112x112/"
    elif args.data_mode == "casia100":
        configuration["DATA_ROOT"] = "./data/faces_webface_112x112_sub100_train_test/"
    elif args.data_mode == "casia1000":
        configuration["DATA_ROOT"] = "./data/faces_webface_112x112_sub1000/"
    elif args.data_mode == "tsne":
        configuration["DATA_ROOT"] = "./data/faces_Tsne_sub/"
    elif args.data_mode == "imagenet100":
        configuration["DATA_ROOT"] = "./data/imagenet100/"
    else:
        raise Exception(args.data_mode)
    configuration["EVAL_PATH"] = "./eval/"
    assert args.net in ["VIT", "VITs", "VIT_B16"]
    configuration["BACKBONE_NAME"] = args.net
    assert args.head in ["Softmax", "ArcFace", "CosFace", "SFaceLoss"]
    configuration["HEAD_NAME"] = args.head
    # configuration['TARGET'] = [i for i in args.target.split(',')]

    if args.resume:
        configuration["BACKBONE_RESUME_ROOT"] = args.resume
    else:
        configuration["BACKBONE_RESUME_ROOT"] = (
            ""  # the root to resume training from a saved checkpoint
        )
    configuration["WORK_PATH"] = args.outdir  # the root to buffer your checkpoints
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # if args has attribute 'evaluate', then only evaluate the model
    configuration["NUM_LAYERS"] = args.vit_depth

    if (
        hasattr(args, "one_stage")
        or hasattr(args, "ewc")
        or hasattr(args, "MAS")
        or hasattr(args, "si")
        or hasattr(args, "online")
        or hasattr(args, "replay")
        or hasattr(args, "l2")
    ):
        configuration["one_stage"] = args.one_stage
        configuration["ewc"] = args.ewc
        configuration["ewc_lambda"] = args.ewc_lambda
        configuration["MAS"] = args.MAS
        configuration["mas_lambda"] = args.mas_lambda
        configuration["si"] = args.si
        configuration["si_c"] = args.si_c
        configuration["online"] = args.online
        configuration["replay"] = args.replay
        configuration["l2"] = args.l2

    if hasattr(args, "BND_pro"):
        configuration["BND_pro"] = args.BND_pro
    if hasattr(args, "few_shot"):
        configuration["few_shot"] = args.few_shot
    
    if hasattr(args, "grouping"):
        configuration["GROUP_TYPE"] = args.grouping

    if hasattr(args, "alpha_epoch"):
        configuration["ALPHA_EPOCH"] = args.alpha_epoch

    if hasattr(args, "per_forget_cls"):
        configuration["PER_FORGET_CLS"] = args.per_forget_cls

    # parameters for LIRF
    if hasattr(args, "LIRF_T"):
        configuration["LIRF_T"] = args.LIRF_T
    if hasattr(args, "LIRF_alpha"):
        configuration["LIRF_alpha"] = args.LIRF_alpha

    # parameter for SCRUB
    configuration["lr_decay_rate"] = 0.1
    if hasattr(args, "scrub_decay_epoch"):
        configuration["lr_decay_epochs"] = args.scrub_decay_epoch
    configuration["sgda_learning_rate"] = args.lr

    # lora pos
    if hasattr(args, "lora_pos"):
        configuration["GROUP_POS"] = args.lora_pos

    return configuration
