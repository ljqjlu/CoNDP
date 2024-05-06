import argparse
import random
import numpy as np
import os.path as osp
import os
import torch
import time
from models.context_neural_process import makecoprocess
from models.models import *
from models.training import DGTimeNeuralProcessTrainer
from torch.utils.data import DataLoader
from data.datasets import *

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
parser.add_argument("--gpu", type=int, default=0, help="which gpu to use (if any)")
parser.add_argument("--data", type=str, choices=["LV", "NS", "GS", "GO"], default="LV", help="which dataset to use")
parser.add_argument("--model", type=str, choices=["np", "ndp"], default="ndp")
parser.add_argument("--epochs", type=int, default=120000) # align with CoDA and LEADS
parser.add_argument("--load", type=eval, choices=[True, False], default=False, help="whether to load an existing model")
parser.add_argument("--batch_size", type=int, default=20, help="Batch size of data loaders")
parser.add_argument("--num_context", type=int, default=15, help="Context size")
parser.add_argument("--num_extra_target", type=int, default=10, help="Maximum target set size")
parser.add_argument("--r_dim", type=int, default=100, help="Dimension of aggregated context")
parser.add_argument("--z_dim", type=int, default=50, help="Dimension of sampled latent variable")
parser.add_argument("--h_dim", type=int, default=100, help="Dimension of hidden states in ODE layers")
parser.add_argument("--L_dim", type=int, default=25, help="Dimension of latent ODE")
parser.add_argument("--use_y0", action="store_true", help="Whether to use y0 or not")
parser.add_argument("--lr", type=float, default=1e-3, help="Model learn rate")
parser.add_argument("--use_all_targets", type=eval, choices=[True, False], default=True,
                    help="Use all points in time series as target")
parser.add_argument("--adaptive_y0", action="store_true", help="Whether to use y0 or not")
parser.add_argument("--adaptive_encoder", action="store_true", help="Whether to use adaptive context encoder or not")
parser.add_argument("--adaptive_decoder", action="store_true", help="Whether to use adaptive latent decoder or not")
parser.add_argument("--disable_disc", action="store_true", help="Disable discriminator")
parser.add_argument("--lamda", type=float, default=0.8)
parser.add_argument("--ratio", type=float, default=0.5, help="Ratio of conditioning data")
parser.add_argument("--ind_prev", type=int, default=20)


def run():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = str(time.time())
    print(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Make folder
    folder = osp.join('results', args.data, args.model, args.exp_name)
    if not osp.exists(folder):
        os.makedirs(folder)
    f = open(osp.join(folder, 'setting.txt'), 'w')
    print(args, file=f)
    f.close()

    print("Loading data...")
    train_data = None
    eval_data = None
    test_cond_data = None
    test_data = None
    if args.data == "LV":
        # dataset = ODEDataset(name="LV", ratio=args.train_ratio, mode="train")
        train_data = ODEDataset(name=args.data, mode="train", ratio=args.ratio)
        eval_data = ODEDataset(name=args.data, mode="train", ratio=args.ratio)
        test_cond_data = ODEDataset(name=args.data, mode="cond", ratio=args.ratio)
        test_data = ODEDataset(name=args.data, mode="test", ratio=args.ratio)
    elif args.data == "NS":
        train_data = ODEDataset(name=args.data, mode="train", ratio=args.ratio)
        eval_data = ODEDataset(name=args.data, mode="train", ratio=args.ratio)
        test_cond_data = ODEDataset(name=args.data, mode="cond", ratio=args.ratio)
        test_data = ODEDataset(name=args.data, mode="test", ratio=args.ratio)
    elif args.data == "GS":
        train_data = ODEDataset(name=args.data, mode="train", ratio=args.ratio)
        eval_data = ODEDataset(name=args.data, mode="train", ratio=args.ratio)
        test_cond_data = ODEDataset(name=args.data, mode="cond", ratio=args.ratio)
        test_data = ODEDataset(name=args.data, mode="test", ratio=args.ratio)
    elif args.data == "GO":
        train_data = ODEDataset(name=args.data, mode="train", ratio=args.ratio)
        eval_data = ODEDataset(name=args.data, mode="train", ratio=args.ratio)
        test_cond_data = ODEDataset(name=args.data, mode="cond", ratio=args.ratio)
        test_data = ODEDataset(name=args.data, mode="test", ratio=args.ratio)
    # print(len(train_data))
    t_min = 0.0
    initial_t = torch.tensor(t_min).view(1, 1, 1).to(device)

    args.n_env = train_data.n_env

    if args.model == "np":
        raise NotImplementedError
    elif args.model == "ndp":
        # daprocess = makedaprocess(dataset=args.data, initial_t=initial_t, adaptive_encoder=args.adaptive_encoder,
        #                           use_y0=args.use_y0, adaptive_y0=args.adaptive_y0,
        #                           adaptive_decoder=args.adaptive_decoder, n_env=args.n_env).to(device)
        # context_net = ContextNet(n_env=args.n_env, context_dim=4, requires_grad=True).to(device)
        # neuralprocess = ContextTimeNeuralProcess(context_net=context_net, encoder=encoder, decoder=decoder)
        # neuralprocess = DGTimeNeuralProcess(daprocess=daprocess, context_net=context_net).to(device)
        neuralprocess = makecoprocess(dataset=args.data, n_env=args.n_env, initial_t=initial_t).to(device)

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    #val_data_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=True)
    test_condition_data_loader = DataLoader(test_cond_data, batch_size=1, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    val_data_loader = test_data_loader

    optimizer = "Adam"
    np_trainer = DGTimeNeuralProcessTrainer(device=device, neural_process=neuralprocess, optim_name=optimizer,
                                            lr=args.lr, context_ratio=args.ratio, use_all_targets=args.use_all_targets,
                                            use_y0=args.use_y0, disable_disc=args.disable_disc, lamda=args.lamda)
    np_trainer.train(train_data_loader, val_data_loader, num_epochs=args.epochs, folder=folder)
    np_trainer.test(len(test_data), test_condition_data_loader, test_data_loader, num_epochs=args.epochs, folder=folder)
    #np_trainer.viz(test_data_loader, folder)


if __name__ == "__main__":
    run()