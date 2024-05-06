import torch
from models.context_neural_process import ContextTimeNeuralProcess
from typing import Tuple
from tqdm import tqdm
from random import randint
import os.path as osp
import time
import numpy as np
from models.utils import context_target_split as cts
from models.utils import test_condition_split as tcs
from torch.distributions.kl import kl_divergence
from torch.distributions import Bernoulli, Normal
from models.models import ContextNet
from datetime import datetime
import copy
import pickle

optimizer_dict = {"Adam": torch.optim.Adam}


class DGTimeNeuralProcessTrainer:
    def __init__(self,
                 device: torch.device,
                 neural_process: ContextTimeNeuralProcess,
                 optim_name: str,
                 context_ratio=0.5,
                 # num_context_range: Tuple[int, int],
                 # num_extra_target_range: Tuple[int, int],
                 lr=1e-3,
                 # num_context_range: Tuple[int, int],
                 # num_extra_target_range: Tuple[int, int],
                 max_context=None,
                 use_all_targets=False,
                 use_y0=True,
                 disable_disc=False,
                 lamda=0.8):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer_dict[optim_name]
        # self.optimizer_A = self.optimizer(filter(lambda p: p.requires_grad, self.neural_process.parameters()), lr=lr)
        # self.neural_process.context_net.set_requires_grad(True)
        # self.optimizer_U = self.optimizer(filter(lambda p: p.requires_grad, self.neural_process.context_net.paramaters()), lr=lr)

        self.optimizer_A = self.optimizer(filter(lambda p: p.requires_grad, self.neural_process.parameters()))
        self.optimizer_U = self.optimizer(
            filter(lambda p: p.requires_grad, self.neural_process.context_net.parameters()))
        if disable_disc:
            self.neural_process.freeze_disc()
            self.optimizer_total = self.optimizer(filter(lambda p: p.requires_grad, self.neural_process.parameters()))
        else:
            self.optimizer_total = self.optimizer(filter(lambda p: p.requires_grad, self.neural_process.parameters()))

        # self.num_context_range = num_context_range
        # self.num_extra_target_range = num_extra_target_range
        self.context_ratio = context_ratio
        self.max_context = max_context
        self.use_all_targets = use_all_targets
        self.use_y0 = use_y0
        self.disable_disc = disable_disc
        self.lamda = lamda

        self.epoch_loss_A_history = []
        self.epoch_loss_U_history = []
        self.epoch_mse_history = []
        self.epoch_nll_history = []
        self.best_model = copy.deepcopy(self.neural_process)

    def optimize_total_epoch(self, data_loader):
        epoch_loss = 0.
        self.neural_process.train()
        # print("Optimizing A")
        for i, data in enumerate(tqdm(data_loader)):
            self.optimizer_total.zero_grad()
            x, y, index = data
            # print()
            if len(y.shape) > 3:
                y = y.view(y.shape[0], y.shape[1], -1)
            points = x.size(1)
            # print(points)

            # num_context_range = tuple((1, int(self.context_ratio * points)))
            num_context_range = tuple((1, int(points)))
            num_context = randint(*num_context_range)
            num_extra_target = points - num_context

            x_context, y_context, x_target, y_target, y0 = (
                cts(x, y, num_context, num_extra_target, use_y0=self.use_y0))
            y0 = y0.to(self.device)
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)
            index = index.to(self.device)

            # time1 = datetime.now()
            p_y_pred, q_target, q_context = self.neural_process(index, x_context, y_context, x_target,
                                                                            y_target, y0)
            # time2 = datetime.now()
            # print("Forward time:", (time2 - time1).seconds)
            loss = self._loss_A(p_y_pred, y_target, q_target, q_context)

            # time1 = datetime.now()
            loss.backward()
            # time2 = datetime.now()
            # print("Backward time:", (time2 - time1).seconds)
            self.optimizer_total.step()

            epoch_loss += loss.cpu().item()

        epoch_loss = epoch_loss / len(data_loader)
        self.epoch_loss_A_history.append(epoch_loss)

        return epoch_loss

    def optimize_A_epoch(self, data_loader):
        epoch_loss = 0.
        self.neural_process.train()
        # print("Optimizing A")
        for i, data in enumerate(tqdm(data_loader)):
            self.optimizer_A.zero_grad()
            x, y, index = data
            points = x.size(1)
            # print(points)

            # num_context_range = tuple((1, int(self.context_ratio * points)))
            num_context_range = tuple((1, int(points)))
            num_context = randint(*num_context_range)
            num_extra_target = points - num_context

            x_context, y_context, x_target, y_target, y0 = (
                cts(x, y, num_context, num_extra_target, use_y0=self.use_y0))
            y0 = y0.to(self.device)
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)
            index = index.to(self.device)

            # time1 = datetime.now()
            p_y_pred, q_target, q_context, p_u_disc, u = self.neural_process(x_context, y_context, index, x_target,
                                                                             y_target, y0)
            # time2 = datetime.now()
            # print("Forward time:", (time2 - time1).seconds)
            loss = self._loss_A(p_y_pred, y_target, q_target, q_context, u, p_u_disc)

            # time1 = datetime.now()
            loss.backward()
            # time2 = datetime.now()
            # print("Backward time:", (time2 - time1).seconds)
            self.optimizer_A.step()

            epoch_loss += loss.cpu().item()

        epoch_loss = epoch_loss / len(data_loader)
        self.epoch_loss_A_history.append(epoch_loss)

        return epoch_loss

    def optimize_U_epoch(self, data_loader):
        epoch_loss = 0.
        self.neural_process.train()
        for i, data in enumerate(tqdm(data_loader)):
            self.optimizer_U.zero_grad()
            x, y, index = data
            points = x.size(1)
            # print(points)

            # num_context_range = tuple((1, int(self.context_ratio * points)))
            num_context_range = tuple((1, int(points)))
            num_context = randint(*num_context_range)
            num_extra_target = points - num_context

            x_context, y_context, x_target, y_target, y0 = (
                cts(x, y, num_context, num_extra_target, use_y0=self.use_y0))
            y0 = y0.to(self.device)
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)
            index = index.to(self.device)

            p_y_pred, q_target, q_context, index_pred = self.neural_process(x_context, y_context, index, x_target,
                                                                            y_target, y0)
            loss = self._loss_U(p_y_pred, y_target, q_target, q_context, index_pred, index)

            loss.backward()
            self.optimizer_U.step()

            epoch_loss += loss.cpu().item()

        epoch_loss = epoch_loss / len(data_loader)
        self.epoch_loss_U_history.append(epoch_loss)

        return epoch_loss

    def train_epoch(self, data_loader):
        # print("train epoch")
        # epoch_loss_U = self.optimize_U_epoch(data_loader)
        epoch_loss = self.optimize_total_epoch(data_loader)
        return epoch_loss
        # epoch_loss_U = self.optimize_U_epoch(data_loader)
        # return epoch_loss_A, epoch_loss_U

    # def train_epoch(self, data_loader):
    #    epoch_loss_A = 0.
    #    epoch_loss_U = 0.
    #    self.neural_process.train()
    #    for i, data in enumerate(tqdm(data_loader)):
    #        self.optimizer_A.zero_grad()
    #        self.optimizer_U.zero_grad()

    #        #x, y, index, mask = data
    #        x, y, index = data
    #        points = x.size(1)

    #        num_context_range = tuple((1, int(self.context_ratio * points)))
    #        num_context = randint(*num_context_range)
    #        #num_context = randint(*self.num_context_range)
    #        num_extra_target = points - num_context
    #        #num_extra_target = randint(*self.num_extra_target_range)
    #        #if self.use_all_targets:
    #        #    num_extra_target = points - num_context

    #        x_context, y_context, x_target, y_target, y0 = (cts(x, y, num_context, num_extra_target, use_y0=self.use_y0))
    #        y0 = y0.to(self.device)
    #        x_context = x_context.to(self.device)
    #        y_context = y_context.to(self.device)
    #        x_target = x_target.to(self.device)
    #        y_target = y_target.to(self.device)
    #        index = index.to(self.device)
    #        #print(y_context.size())
    #        #u = neural_process.context_net(index)

    #        p_y_pred, q_target, q_context, p_u_disc, u = self.neural_process(x_context, y_context, index, x_target, y_target, y0)
    #        loss_A, loss_U = self._loss(p_y_pred, y_target, q_target, q_context, u, p_u_disc)

    #        loss_A.backward(retain_graph=True)
    #        self.optimizer_A.step()

    #        #with torch.autograd.set_detect_anomaly(True):
    #        #    loss_U.backward()
    #        #    self.optimizer_U.step()

    #        epoch_loss_A += loss_A.cpu().item()
    #        epoch_loss_U += loss_U.cpu().item()

    #    epoch_loss_A = epoch_loss_A / len(data_loader)
    #    epoch_loss_U = epoch_loss_U / len(data_loader)
    #    self.epoch_loss_A_history.append(epoch_loss_A)
    #    self.epoch_loss_U_history.append(epoch_loss_U)

    #    return epoch_loss_A, epoch_loss_U

    def eval_epoch(self, data_loader, context_size=None):
        epoch_mse = 0
        epoch_nll = 0
        epoch_mape = 0
        # if context_size is None:
        #    context_size = randint(*self.num_context_range)

        self.neural_process.eval()
        for i, data in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                # x, y, index, mask = data
                x, y, index = data
                if len(y.shape) > 3:
                    y = y.view(y.shape[0], y.shape[1], -1)
                points = x.size(1)
                num_context_range = tuple((1, int(points)))
                context_size = randint(*num_context_range)
                # num_context_range = tuple((1, int(self.context_ratio * points)))
                # context_size =  int(self.context_ratio * points)
                # locations = list(range(int(self.context_ratio * points)))
                x_context, y_context, _, _, y0 = cts(x, y, context_size, 0, use_y0=self.use_y0)

                y0 = y0.to(self.device)
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)

                x_target = x.to(self.device)
                y_target = y.to(self.device)
                index = index.to(self.device)
                p_y_pred = self.neural_process(index, x_context, y_context, x_target, y_target, y0)

                nll = self._loss_U(p_y_pred, y_target)
                epoch_nll += nll.cpu().item()

                mse = ((y_target - p_y_pred.mean) ** 2).mean()
                epoch_mse += mse.item()

                raw_mape = torch.abs(y_target - p_y_pred.mean) / torch.abs(y_target)
                mape = torch.mean(raw_mape[(~raw_mape.isnan()) & raw_mape.isfinite()])
                epoch_mape += mape.cpu().item()

        epoch_mse = epoch_mse / len(data_loader)
        epoch_nll = epoch_nll / len(data_loader)
        epoch_mape = epoch_mape / len(data_loader)

        self.epoch_mse_history.append(epoch_mse)
        self.epoch_nll_history.append(epoch_nll)

        return epoch_mse, epoch_nll, epoch_mape

    def test_optim_epoch(self, data_loader):
        epoch_loss_U = self.optimize_U_epoch(data_loader)
        return epoch_loss_U

    # def test_optim_epoch(self, data_loader):
    #    #epoch_mse = 0
    #    #epoch_nll = 0
    #    for i, data in enumerate(tqdm(data_loader)):
    #        self.optimizer_U.zero_grad()
    #        x, y, index = data
    #        points = x.size(1)

    #        num_context = randint(*self.num_context_range)
    #        x_context, y_context, _, _, y0 = cts(x, y, num_context, 0, use_y0=self.use_y0)

    #        #x_condition, y_condition, y0 =

    #        y0 = y0.to(self.device)
    #        x_context = x_context.to(self.device)
    #        y_context = y_context.to(self.device)

    #        #x_target = x_context.to(self.device)
    #        #y_target = y_context.to(self.device)
    #        x_target = x.to(self.device)
    #        y_target = y.to(self.device)
    #        index = index.to(self.device)

    #        p_y_pred, q_target, q_context, p_u_disc, u = self.neural_process(x_context, y_context, index, x_target, y_target, y0)
    #        loss_A, loss_U = self._loss(p_y_pred, y_target, q_target, q_context, u, p_u_disc)

    #        loss_U.backward()
    #        self.optimizer_U.step()

    # def test(self, n_env_test, test_condition_data_loader, test_prediction_data_loader, num_epochs, folder):
    # self.neural_process = copy.deepcopy(self.best_model)
    # context_net = ContextNet(n_env_test, self.neural_process.context_net.context_dim, self.neural_process.context_net.requires_grad).to(self.device)
    # self.neural_process.context_net = context_net
    # self.freeze(self.neural_process.daprocess)
    # self.optimizer_U = self.optimizer(filter(lambda p: p.requires_grad, self.neural_process.context_net.parameters()))
    # best_test_nll = 1e15
    # start_time = time.time()
    # for epoch in range(int(3 * num_epochs)):
    # optim_nll = self.test_optim_epoch(test_condition_data_loader)
    # if optim_nll < best_test_nll:
    # best_test_nll = optim_nll
    # self.best_model = copy.deepcopy(self.neural_process)
    # torch.save(self.neural_process.state_dict(), osp.join(folder, "test_model.pth"))
    # print("Epoch: " + str(epoch) + " | Optim NLL: " + str(optim_nll) + "| Best test NLL: " + str(best_test_nll))
    # self.neural_process = copy.deepcopy(self.best_model)
    # test_mse, test_nll = self.test_epoch(test_prediction_data_loader)
    # print("Test MSE: " + str(test_mse) + "| Test NLL: " + str(test_nll))
    # np.save(osp.join(folder, 'test_mse.npy'), np.array([test_mse]))
    # np.save(osp.join(folder, 'test_nll.npy'), np.array([test_nll]))
    # return test_mse, test_nll

    def test(self, n_env_test, test_condition_data_loader, test_prediction_data_loader, num_epochs, folder):
        # self.neural_process = copy.deepcopy(self.best_model)
        # epoch_mse = 0
        # epoch_nll = 0
        # epoch_mape = 0
        # test_num_epochs = 100
        # for epoch in range(int(test_num_epochs)):
        # test_mse, test_nll, test_mape = self.test_epoch(test_prediction_data_loader)
        # epoch_mse += test_mse
        # epoch_nll += test_nll
        # epoch_mape += test_mape
        # print("Epoch: " + str(epoch) + " | Test MSE: " + str(test_mse) + "| Test NLL: " + str(test_nll) + "| Test MAPE: " + str(test_mape))
        # epoch_mse = epoch_mse / int(test_num_epochs)
        # epoch_nll = epoch_nll / int(test_num_epochs)
        # epoch_mape = epoch_mape / int(test_num_epochs)
        # print("Total Test MSE: " + str(epoch_mse) + "| Total Test NLL: " + str(epoch_nll) + "| Total Test MAPE: " + str(epoch_mape))

        # self.neural_process = copy.deepcopy(self.best_model)
        # context_net = ContextNet(n_env_test, self.neural_process.context_net.context_dim, self.neural_process.context_net.requires_grad).to(self.device)
        # self.neural_process.context_net = context_net
        # self.freeze(self.neural_process.daprocess)
        # self.optimizer_U = self.optimizer(filter(lambda p: p.requires_grad, self.neural_process.context_net.parameters()))
        # best_test_nll = 1e15
        # start_time = time.time()
        # for epoch in range(int(100)):
        # optim_nll = self.test_optim_epoch(test_condition_data_loader)
        # if optim_nll < best_test_nll:
        # best_test_nll = optim_nll
        # self.best_model = copy.deepcopy(self.neural_process)
        # torch.save(self.neural_process.state_dict(), osp.join(folder, "test_model.pth"))
        # print("Epoch: " + str(epoch) + " | Optim NLL: " + str(optim_nll) + "| Best test NLL: " + str(best_test_nll))

        self.neural_process = copy.deepcopy(self.best_model)
        epoch_mse = 0
        epoch_nll = 0
        epoch_mape = 0
        test_num_epochs = 100
        for epoch in range(int(test_num_epochs)):
            test_mse, test_nll, test_mape = self.test_epoch(test_prediction_data_loader)
            epoch_mse += test_mse
            epoch_nll += test_nll
            epoch_mape += test_mape
            print("Epoch: " + str(epoch) + " | Test MSE: " + str(test_mse) + "| Test NLL: " + str(
                test_nll) + "| Test MAPE: " + str(test_mape))
        epoch_mse = epoch_mse / int(test_num_epochs)
        epoch_nll = epoch_nll / int(test_num_epochs)
        epoch_mape = epoch_mape / int(test_num_epochs)
        np.save(osp.join(folder, 'test_mse.npy'), np.array([epoch_mse]))
        np.save(osp.join(folder, 'test_nll.npy'), np.array([epoch_nll]))
        np.save(osp.join(folder, 'test_mape.npy'), np.array([epoch_mape]))
        return epoch_mse, epoch_nll

    def test_epoch(self, data_loader, context_size=None):
        epoch_mse = 0
        epoch_nll = 0
        epoch_mape = 0

        self.neural_process.eval()
        for i, data in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                x, y, index = data
                if len(y.shape) > 3:
                    y = y.view(y.shape[0], y.shape[1], -1)
                points = x.size(1)
                context_size = int(self.context_ratio * points)
                x_context, y_context, _, _, y0 = cts(x, y, context_size, 0, use_y0=self.use_y0)
                # x_context, y_context, _, _, y0 = cts(x, y, 1, 0, use_y0=self.use_y0)

                y0 = y0.to(self.device)
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)

                x_target = x.to(self.device)
                y_target = y.to(self.device)
                index = index.to(self.device)
                p_y_pred = self.neural_process(index, x_context, y_context, x_target, y_target, y0)

                nll = self._loss_U(p_y_pred, y_target)
                epoch_nll += nll.cpu().item()

                mse = ((y_target - p_y_pred.mean) ** 2).mean()
                epoch_mse += mse.cpu().item()

                raw_mape = torch.abs(y_target - p_y_pred.mean) / torch.abs(y_target)
                mape = torch.mean(raw_mape[(~raw_mape.isnan()) & raw_mape.isfinite()])
                epoch_mape += mape.cpu().item()

        epoch_mse = epoch_mse / len(data_loader)
        epoch_nll = epoch_nll / len(data_loader)
        epoch_mape = epoch_mape / len(data_loader)

        return epoch_mse, epoch_nll, epoch_mape

    def train(self, train_data_loader, val_data_loader, num_epochs, folder):
        # Optimize on train loader and evaluation on val loader
        # Train loader and val loader share the same index
        best_eval_nll = 1e15
        # best_eval_mse = 1e15
        best_model = None
        start_time = time.time()
        mse_list = []
        mape_list = []
        # self.neural_process.train()
        for epoch in range(num_epochs):
            # train_loss_A, train_loss_U = self.train_epoch(train_data_loader)
            train_loss_A = self.train_epoch(train_data_loader)
            val_mse, val_nll, val_mape = self.eval_epoch(val_data_loader)

            mse_list.append(val_mse)
            mape_list.append(val_mape)

            if val_nll < best_eval_nll:
                best_eval_nll = val_nll
                self.best_model = copy.deepcopy(self.neural_process)
                #    best_eval_mse = val_mse
                torch.save(self.neural_process.state_dict(), osp.join(folder, "model.pth"))

            print("Epoch: " + str(epoch) + "| Train loss Ad: " + str(train_loss_A) + "|Validation MSE: " + str(
                val_mse) + "| Val NLL: " + str(val_nll))

            # print("Epoch: " + str(epoch) + "| Train loss Ad: " + str(train_loss_A) + "| Train loss U: " + str(train_loss_U) + "|Validation MSE: " + str(val_mse) + "| Val NLL: " + str(val_nll))

        end_time = time.time()
        # self.ind_mse = ind_mse
        # self.ind_mse_std = ind_mse_std
        # self.ind_mape = ind_mape
        # self.ind_mape_std = ind_mape_std
        np.save(osp.join(folder, 'training_time.npy'), np.array([end_time - start_time]))
        np.save(osp.join(folder, 'loss_history_ad.npy'), np.array(self.epoch_loss_A_history))
        np.save(osp.join(folder, 'loss_history_context.npy'), np.array(self.epoch_loss_U_history))
        np.save(osp.join(folder, 'eval_mse_history.npy'), np.array(self.epoch_mse_history))
        np.save(osp.join(folder, 'eval_logp_history.npy'), np.array(self.epoch_nll_history))
        print(f"Total time = {end_time - start_time}")

    def viz(self, test_prediction_data_loader, folder):
        self.neural_process = copy.deepcopy(self.best_model)
        self.neural_process.eval()
        viz = []
        for i, data in enumerate(tqdm(test_prediction_data_loader)):
            viz_dict = dict()
            with torch.no_grad():
                x, y, index = data
                if len(y.shape) > 3:
                    y = y.view(y.shape[0], y.shape[1], -1)
                points = x.size(1)
                context_size = int(self.context_ratio * points)
                x_context, y_context, _, _, y0 = cts(x, y, context_size, 0, use_y0=self.use_y0)
                # x_context, y_context, _, _, y0 = cts(x, y, 1, 0, use_y0=self.use_y0)

                y0 = y0.to(self.device)
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)

                x_target = x.to(self.device)
                y_target = y.to(self.device)
                index = index.to(self.device)
                p_y_pred = self.neural_process(index, x_context, y_context, x_target, y_target, y0)
                viz_dict["x"] = x.cpu()
                viz_dict["y"] = y.cpu()
                viz_dict["x_context"] = x_context.cpu()
                viz_dict["y_context"] = y_context.cpu()
                viz_dict["y_pred"] = p_y_pred.mean.cpu()
                viz_dict["env"] = index.cpu()
                viz.append(viz_dict)

        with open(osp.join(folder, "viz_ndp.pkl"), "wb") as f:
            pickle.dump(viz, f)
        f.close()

    # def test_epoch(self, data_loader):

    # def test(self, n_env_test, test_condition_data_loader, test_prediction_data_loader, num_epochs, folder):
    # self.neural_process = copy.deepcopy(self.best_model)
    # best_test_nll = 1e15
    # start_time = time.time()
    # epoch_mse = 0
    # epoch_nll = 0
    # test_num_epochs = 100
    # for epoch in range(int(test_num_epochs)):
    # test_mse, test_nll = self.test_epoch(test_prediction_data_loader)
    # epoch_mse += test_mse
    # epoch_nll += test_nll
    # print("Epoch: " + str(epoch) + " | Test MSE: " + str(test_mse) + "| Test NLL: " + str(test_nll))
    # epoch_mse = epoch_mse / int(test_num_epochs)
    # epoch_nll = epoch_nll / int(test_num_epochs)
    # print("Total Test MSE: " + str(epoch_mse) + "| Total Test NLL: " + str(epoch_nll))
    # np.save(osp.join(folder, 'test_mse.npy'), np.array([epoch_mse]))
    # np.save(osp.join(folder, 'test_nll.npy'), np.array([epoch_nll]))
    # return epoch_mse, epoch_nll

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def _loss_A(self, p_y_pred, y_target, q_target=None, q_context=None):
        if isinstance(p_y_pred, Bernoulli):
            pred = p_y_pred.logits
            loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            nll = loss(pred, y_target).mean()
        else:
            nll = -p_y_pred.log_prob(y_target).mean()
        #return nll

        if q_target is None and q_context is None:
            return nll

        kl = kl_divergence(q_target, q_context).mean()

        return nll + kl

    def _loss_U(self, p_y_pred, y_target, q_target=None, q_context=None, index_pred=None, index=None):
        if isinstance(p_y_pred, Bernoulli):
            pred = p_y_pred.logits
            loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            nll = loss(pred, y_target).mean()
        else:
            nll = -p_y_pred.log_prob(y_target).mean()

        return nll

    def _loss(self, p_y_pred, y_target, q_target=None, q_context=None, u=None, p_u_disc=None):
        if isinstance(p_y_pred, Bernoulli):
            pred = p_y_pred.logits
            loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            nll = loss(pred, y_target).mean()
        else:
            nll = -p_y_pred.log_prob(y_target).mean()

        loss_U = nll
        loss_A = nll

        if q_target is None and q_context is None:
            loss_A += 0
        else:
            kl = kl_divergence(q_target, q_context).mean()
            loss_A += kl
            if self.disable_disc:
                loss_A += 0
            else:
                nll_disc = -p_u_disc.log_prob(u).mean()
                loss_A += self.lamda * nll_disc

        return loss_A, loss_U