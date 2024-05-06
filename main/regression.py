import argparse
import os
import os.path as osp
import time
from data.data import LotkaVolterraDataset
from torch.utils.data import DataLoader
from models.context_neural_process import *

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
# parser.add_argument("--model", type=str, )
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--data", type=str, choices=["deterministic_lv"], default="deterministic_lv")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--load", type=eval, choices=[True, False], default=False)
args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run():
    device = torch.device(f'cuda: {args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(device)

    folder = osp.join('results/context', args.data, args.exp_name)
    if not osp.exists(folder):
        os.makedirs(folder)

    print("Creating Data")
    if args.data == "deterministic_lv":
        dataset_train_params = {
            "n_data_per_env": 4, "t_horizon": 20, "dt": 0.5, "method": "RK45",
            "params": [
                {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
                {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
                {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
                {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0}
            ]
        }
        dataset_test_params = dict()
        dataset_test_params.update(dataset_train_params)
        dataset_test_params["n_data_per_env"] = 32
        dataset_train, dataset_test = LotkaVolterraDataset(**dataset_train_params), LotkaVolterraDataset(
            **dataset_test_params)

    n_env = len(dataset_train_params["params"])
    batch_size = 36
    dataloader_train, dataloader_test = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                                   drop_last=False), DataLoader(dataset=dataset_train,
                                                                                batch_size=batch_size, drop_last=False)
