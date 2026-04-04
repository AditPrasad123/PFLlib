#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import torch.nn as nn

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverbabu import FedBABU

from flcore.trainmodel.models import BaseHeadSplit

from quanv_efficientnet_b0 import QuanvEfficientNetB0Improved
from quanv_tinyvit import QuanvTinyViTImproved
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        
        if "_quanv" in args.dataset:
            if model_str == "EfficientNetB0":
                print(f"⚠️  Auto-switching from EfficientNetB0 to QuanvEfficientNetB0 for {args.dataset} dataset")
                model_str = "QuanvEfficientNetB0"
            elif model_str == "TinyViT":
                print(f"⚠️  Auto-switching from TinyViT to QuanvTinyViT for {args.dataset} dataset")
                model_str = "QuanvTinyViT"
        
        # Generate args.model
        
        if model_str == "QuanvEfficientNetB0":
            args.model = QuanvEfficientNetB0Improved(
                num_classes=args.num_classes, 
                pretrained=True, 
                improvement_level='standard'  # Use the improved version with better performance
                ).to(args.device)

        elif model_str == "QuanvTinyViT":
            args.model = QuanvTinyViTImproved(
                num_classes=args.num_classes,
                pretrained=True,
                improvement_level='standard'  # Use the improved version with better performance
            ).to(args.device)
            
        elif model_str == "EfficientNetB0":
            model = torchvision.models.efficientnet_b0(pretrained=True)

            in_features = model.classifier[1].in_features

            # Replace classifier with identity
            model.classifier[1] = nn.Identity()

            # Create fc manually (so framework works)
            model.fc = nn.Linear(in_features, args.num_classes)

            args.model = model.to(args.device)

        elif model_str == "TinyViT":
            # Load the Tiny-ViT model hosted on Hugging Face via timm
            try:
                import timm
            except Exception:
                raise ImportError("Please install timm and huggingface-hub: pip install timm huggingface-hub")

            # model name on HF: 'timm/tiny_vit_5m_224.dist_in22k'
            # timm accepts the model name (without 'timm/') for create_model
            model_name = 'tiny_vit_5m_224.dist_in22k'
            # Create model without forcing classifier; we'll reset appropriately
            model = timm.create_model(model_name, pretrained=True)

            # Get feature dimension from forward_features
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = model.forward_features(dummy_input)
                nf = features.numel() // features.shape[0]  # total features per sample

            # Always use a simple linear classifier for better performance
            model.head = nn.Identity()  # Make base return features
            model.fc = nn.Linear(nf, args.num_classes)

            args.model = model.to(args.device)
            
        else:
            raise NotImplementedError

        print(args.model)
        
        # Store model name string for result file naming
        args.model_name = model_str

        # Optionally freeze backbone weights and train only the classification head
        if getattr(args, 'freeze_backbone', False):
            print("Freezing backbone parameters; only the head will be trained.")
            # Identify head attribute (common names)
            head_attr = None
            for name in ('fc', 'head', 'classifier'):
                if hasattr(args.model, name):
                    head_attr = getattr(args.model, name)
                    break

            # Freeze all params
            for p in args.model.parameters():
                p.requires_grad = False

            # Unfreeze head params if found
            if head_attr is not None:
                for p in head_attr.parameters():
                    p.requires_grad = True
            else:
                print('Warning: could not find model head attribute to unfreeze (expected fc/head/classifier).')

        # select algorithm
        if args.algorithm == "FedAvg":
            # Models with built-in base/head split should not be wrapped again.
            if model_str not in ["QuanvTinyViT", "QuanvEfficientNetB0"]:
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedBABU":
            # Skip BaseHeadSplit wrapping for models that already have internal base/head split
            if model_str not in ["QuanvTinyViT", "QuanvEfficientNetB0", "EfficientNetB0Kernel"]:
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
            else:
                # Models with built-in base/head split (VQCHybrid, Hybrid, QuanvTinyViT, QuanvEfficientNetB0, InceptionV3)
                # Store head for reference but don't wrap the model
                args.head = args.model.head
            server = FedBABU(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    model_name = getattr(args, 'model_name', '')
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times, model=model_name, prev=args.prev)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=-1,
                        help="Number of classes. If -1, auto-detect from dataset config.")
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # imbalance handling
    parser.add_argument('-cwl', "--class_weighted_loss", type=bool, default=False,
                        help="Use class-weighted CrossEntropyLoss per client")
    parser.add_argument('-fl', "--focal_loss", type=bool, default=False,
                        help="Use FocalLoss instead of CrossEntropyLoss")
    parser.add_argument('-fg', "--focal_gamma", type=float, default=2.0,
                        help="Gamma for FocalLoss")
    parser.add_argument('-lsm', "--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for CrossEntropyLoss (0-0.2 typical)")
    parser.add_argument('-os', "--oversample", type=bool, default=False,
                        help="Use WeightedRandomSampler to balance classes per client")
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # freeze pretrained backbone and train only the head
    parser.add_argument('-fb', "--freeze_backbone", type=bool, default=False,
                        help="Freeze pretrained backbone weights and train only the head")
    
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    # Auto-detect num_classes from dataset config if not specified
    if args.num_classes == -1:
        import json
        config_path = os.path.join('../dataset', args.dataset, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                args.num_classes = config.get('num_classes', 10)
                args.num_clients = config.get('num_clients', args.num_clients)
                print(f"Auto-detected num_classes={args.num_classes}, num_clients={args.num_clients} from {args.dataset} config.")
        else:
            args.num_classes = 10
            print(f"Config not found; using default num_classes=10.")
    else:
        # Auto-detect num_clients from config even if num_classes is specified
        import json
        config_path = os.path.join('../dataset', args.dataset, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                args.num_clients = config.get('num_clients', args.num_clients)
                print(f"Auto-detected num_clients={args.num_clients} from {args.dataset} config.")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")