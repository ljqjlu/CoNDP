import os.path as osp
import pickle
import numpy as np
import torch
import random
from tqdm import tqdm
import os
from scipy.integrate import solve_ivp
from functools import partial
import torch
import math
import shelve
import sys
import argparse


class ODEDataGenerator():
    def __init__(self, name, n_env, n_data_per_env, t_horizon, dt, method="RK45", n_env_cond=None,
                 n_data_cond_per_env=None, n_data_pred_per_env=None, ratio=0.5):
        super(ODEDataGenerator, self).__init__()
        tgt_path = osp.join(os.getcwd(), f"datasets/{name}")
        self.file_path = osp.join(tgt_path, f"ratio{ratio}")
        self.n_env = n_env
        self.n_data_per_env = n_data_per_env
        self.n_env_cond = n_env_cond
        self.n_data_cond_per_env = n_data_cond_per_env
        self.n_data_pred_per_env = n_data_pred_per_env
        self.t_horizon = t_horizon
        self.dt = dt
        self.method = method
        self.envs_train = self.create_envs(mode="train")
        self.envs_test = self.create_envs(mode="test")
        self.ratio = ratio

    def generate_data(self):
        meta_dict = dict()
        meta_dict["envs_train"] = self.envs_train
        meta_dict["envs_test"] = self.envs_test
        with open(osp.join(self.file_path, "meta_data.pkl"), "wb") as f:
            pickle.dump(meta_dict, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env_cond)):
            env = self.envs_test[i]
            for j in tqdm(range(self.n_data_cond_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                points = np.arange(times.shape[0])
                size = int(self.ratio * times.shape[0])
                if size < 1:
                    size = 1
                initial_loc = np.array([0])
                size -= 1
                points = points[1:]
                locations = np.random.choice(points, size=size, replace=False)
                locations = np.concatenate([initial_loc, locations])
                locations = np.sort(locations)
                times = times[locations, :]
                states = states[locations, :]
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_cond.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env_cond)):
            env = self.envs_test[i]
            for j in tqdm(range(self.n_data_pred_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_test.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env)):
            env = self.envs_train[i]
            for j in tqdm(range(self.n_data_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                points = np.arange(times.shape[0])
                size = int(self.ratio * times.shape[0])
                if size < 1:
                    size = 1
                initial_loc = np.array([0])
                size -= 1
                points = points[1:]
                locations = np.random.choice(points, size=size, replace=False)
                locations = np.concatenate([initial_loc, locations])
                locations = np.sort(locations)
                times = times[locations, :]
                states = states[locations, :]
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_train.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env_cond)):
            env = self.envs_train[i]
            for j in tqdm(range(self.n_data_pred_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_eval.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

    def generate_ts(self, env):
        y0 = self._get_init_cond()
        # env = self.envs[index]
        # print(env)
        times = np.arange(0., self.t_horizon, self.dt)
        # print(partial(self._f, env=env))
        y = solve_ivp(partial(self._f, env=env), (0., self.t_horizon), y0=y0, method=self.method, t_eval=times)
        states = torch.from_numpy(y.y).float()
        states = torch.transpose(states, 0, 1)
        times = torch.from_numpy(times).float()
        times = times.unsqueeze(-1)
        return times, env, states

    def create_envs(self, mode="train"):
        envs = []
        for i in range(self.n_env):
            envs.append(self.create_env(mode))
        return envs

    def create_env(self, mode="train"):
        raise NotImplementedError

    def _get_init_cond(self):
        raise NotImplementedError

    def _get_env(self):
        raise NotImplementedError

    def _f(self, t, x, env):
        raise NotImplementedError

    # def _observe_env(self, )
    def is_legal(self, times, states):
        raise NotImplementedError


class LVGenerator(ODEDataGenerator):
    def _get_init_cond(self):
        return np.random.uniform(low=1.0, high=3.0, size=(2))

    def create_env(self, mode="train"):
        env = dict()
        if mode in ["train"]:
            env["alpha"] = random.uniform(0.25, 1.25)
            env["beta"] = random.uniform(0.25, 1.25)
            env["gamma"] = random.uniform(0.25, 1.25)
            env["delta"] = random.uniform(0.25, 1.25)

            params = list(env.values())
            if all(((p >= 0.55 and p < 0.95) for p in params)):
                k = random.choices(list(env.keys()))
                p = random.uniform(0.25, 1.25)
                while (0.55 <= p < 0.95):
                    p = random.uniform(0.25, 1.25)
                env[k[0]] = p

        else:
            env["alpha"] = random.uniform(0.55, 0.95)
            env["beta"] = random.uniform(0.55, 0.95)
            env["gamma"] = random.uniform(0.55, 0.95)
            env["delta"] = random.uniform(0.55, 0.95)
        return env

    def _f(self, t, x, env):
        # print(env)
        alpha = env["alpha"]
        beta = env["beta"]
        gamma = env["gamma"]
        delta = env["delta"]
        d = np.zeros(2)
        d[0] = alpha * x[0] - beta * x[0] * x[1]
        d[1] = delta * x[0] * x[1] - gamma * x[1]
        return d

    def is_legal(self, times, states):
        return (not np.isnan(states).any()) and np.isfinite(states).all() and (states >= 0.0).all()


class GOGenerator(ODEDataGenerator):
    def _get_init_cond(self):
        ic_range = [(0.15, 1.60), (0.19, 2.16), (0.04, 0.20), (0.10, 0.35), (0.08, 0.30), (0.14, 2.67), (0.05, 0.10)]
        return np.random.random(7) * np.array([b - a for a, b in ic_range]) + np.array([a for a, _ in ic_range])

    def create_env(self, mode="train"):
        env = dict()
        value_total = {"J0": (2.2, 2.8), "k1": (85, 115), "k2": (4.5, 7.5), "k3": (13, 19), "k4": (85, 115),
                       "k5": (1.13, 1.43), "k6": (10.5, 13.5), "K1": (0.45, 1.05), "q": (3.4, 4.6), "N": (0.4, 1.6),
                       "A": (3.4, 4.6), "kappa": (10, 16), "psi": (0.04, 0.16), "k": (1.5, 2.1)}
        value_test = {"J0": (2.4, 2.6), "k1": (95, 105), "k2": (5.5, 6.5), "k3": (15, 17), "k4": (95, 105),
                      "k5": (1.23, 1.33), "k6": (11.5, 12.5), "K1": (0.65, 0.85), "q": (3.8, 4.2), "N": (0.8, 1.2),
                      "A": (3.8, 4.2), "kappa": (12, 14), "psi": (0.08, 0.12), "k": (1.7, 1.9)}
        if mode in ["train"]:
            for key in value_total.keys():
                env[key] = random.uniform(*value_total[key])

            flag = True
            for param in value_test.keys():
                if not (value_test[param][0] <= env[param] <= value_test[param][1]):
                    flag = False
                    break

            if flag:
                k = random.choices(list(env.keys()))[0]
                p = random.uniform(*value_total[k])
                while (value_test[param][0] <= p <= value_test[param][1]):
                    p = random.uniform(*value_total[k])
                env[k] = p

        else:
            for key in value_total.keys():
                env[key] = random.uniform(*value_test[key])
        return env

    def _f(self, t, x, env):
        keys = ['J0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'K1', 'q', 'N', 'A', 'kappa', 'psi', 'k']
        J0, k1, k2, k3, k4, k5, k6, K1, q, N, A, kappa, psi, k = [env[k] for k in keys]

        d = np.zeros(7)
        k1s1s6 = k1 * x[0] * x[5] / (1 + (x[5] / K1) ** q)
        d[0] = J0 - k1s1s6
        d[1] = 2 * k1s1s6 - k2 * x[1] * (N - x[4]) - k6 * x[1] * x[4]
        d[2] = k2 * x[1] * (N - x[4]) - k3 * x[2] * (A - x[5])
        d[3] = k3 * x[2] * (A - x[5]) - k4 * x[3] * x[4] - kappa * (x[3] - x[6])
        d[4] = k2 * x[1] * (N - x[4]) - k4 * x[3] * x[4] - k6 * x[1] * x[4]
        d[5] = -2 * k1s1s6 + 2 * k3 * x[2] * (A - x[5]) - k5 * x[5]
        d[6] = psi * kappa * (x[3] - x[6]) - k * x[6]
        return d

    def is_legal(self, times, states):
        return (not np.isnan(states).any()) and np.isfinite(states).all() and (states >= 0.0).all()


class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic"):
        self.dim = dim
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1), torch.arange(start=-k_max, end=0, step=1)), 0)
            self.sqrt_eig = size * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.
        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, 1)
            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers
            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                    (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0
        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, size, 1)
            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)
            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                    (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self):
        coeff = torch.randn(*self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.ifftn(coeff)
        u = u.real
        return u


class NSGenerator(object):
    def __init__(self, name, n_env, n_data_per_env, t_horizon, dt_eval, method="RK45", n_env_cond=None,
                 n_data_cond_per_env=None, n_data_pred_per_env=None, size=32, dx=2., buffer_file=None, ratio=0.5):
        super(NSGenerator, self).__init__()
        tgt_path = osp.join(os.getcwd(), f"datasets/{name}")
        self.file_path = osp.join(tgt_path, f"ratio{ratio}")
        self.n_env = n_env
        self.n_data_per_env = n_data_per_env
        self.n_env_cond = n_env_cond
        self.n_data_cond_per_env = n_data_cond_per_env
        self.n_data_pred_per_env = n_data_pred_per_env
        self.t_horizon = t_horizon
        # self.dt = dt
        self.method = method
        self.size = int(size)  # size of the 2D grid
        tt = torch.linspace(0, 1, size + 1)[0:-1]
        X, Y = torch.meshgrid(tt, tt)
        self.forcing_zero = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
        self.envs_train = self.create_envs(mode="train")
        self.envs_test = self.create_envs(mode="test")
        self.n_data_per_env = n_data_per_env
        # self.num_env = len(params)
        # self.len = n_data_per_env * self.num_env
        self.dx = dx  # space step discretized domain [-1, 1]
        self.t_horizon = float(t_horizon)  # total time
        self.n = int(t_horizon / dt_eval)  # number of iterations
        self.sampler = GaussianRF(2, self.size, alpha=2.5, tau=7)
        self.dt_eval = dt_eval
        self.dt = 1e-3
        # self.buffer = shelve.open(buffer_file)
        # self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.method = method
        self.ratio = ratio
        # self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]

    def navier_stokes_2d(self, w0, env, T, delta_t, record_steps):
        alpha = env["alpha"]
        alpha1 = env["alpha1"]
        alpha2 = env["alpha2"]
        beta = env["beta"]
        beta1 = env["beta1"]
        beta2 = env["beta2"]
        tt = torch.linspace(0, 1, self.size + 1)[0:-1]
        X, Y = torch.meshgrid(tt, tt)
        f = alpha * torch.sin(2 * math.pi * (alpha1 * X + alpha2 * Y)) + beta * torch.cos(
            2 * math.pi * (beta1 * X + beta2 * Y))
        visc = env["visc"]
        # Grid size - must be power of 2
        N = w0.size()[-1]
        # Maximum frequency
        k_max = math.floor(N / 2.0)
        # Number of steps to final time
        steps = math.ceil(T / delta_t)
        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, (N, N))
        # Forcing to Fourier space
        f_h = torch.fft.fftn(f, (N, N))
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
        # Record solution every this number of steps
        record_time = math.floor(steps / record_steps)
        # Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                         torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
        # Wavenumbers in x-direction
        k_x = k_y.transpose(0, 1)
        # Negative Laplacian in Fourier space
        lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap[0, 0] = 1.0
        # Dealiasing mask
        dealias = torch.unsqueeze(
            torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)
        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, 1, device=w0.device, dtype=torch.float)
        sol_t = torch.zeros(record_steps, device=w0.device)
        # Record counter
        c = 0
        # Physical time
        t = 0.0
        for j in range(steps):
            if j % record_time == 0:
                # Solution in physical space
                w = torch.fft.ifftn(w_h, (N, N))
                # Record solution and time
                sol[..., c, 0] = w.real
                # sol[...,c,1] = w.imag
                sol_t[c] = t
                c += 1
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h.clone()
            psi_h = psi_h / lap
            # Velocity field in x-direction = psi_y
            q = psi_h.clone()
            temp = q.real.clone()
            q.real = -2 * math.pi * k_y * q.imag
            q.imag = 2 * math.pi * k_y * temp
            q = torch.fft.ifftn(q, (N, N))
            # Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            temp = v.real.clone()
            v.real = 2 * math.pi * k_x * v.imag
            v.imag = -2 * math.pi * k_x * temp
            v = torch.fft.ifftn(v, (N, N))
            # Partial x of vorticity
            w_x = w_h.clone()
            temp = w_x.real.clone()
            w_x.real = -2 * math.pi * k_x * w_x.imag
            w_x.imag = 2 * math.pi * k_x * temp
            w_x = torch.fft.ifftn(w_x, (N, N))
            # Partial y of vorticity
            w_y = w_h.clone()
            temp = w_y.real.clone()
            w_y.real = -2 * math.pi * k_y * w_y.imag
            w_y.imag = 2 * math.pi * k_y * temp
            w_y = torch.fft.ifftn(w_y, (N, N))
            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q * w_x + v * w_y, (N, N))
            # Dealias
            F_h = dealias * F_h
            # Cranck-Nicholson update
            w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / \
                  (1.0 + 0.5 * delta_t * visc * lap)
            # Update real time (used only for recording)
            t += delta_t

        return sol, sol_t

    def _get_init_cond(self):
        w0 = self.sampler.sample()
        return w0
        # state, _ = self.navier_stokes_2d(w0, f=self.forcing_zero, visc=8e-4, T=30.0,
        # delta_t=self.dt, record_steps=20)
        # init_cond = state[:, :, -1, 0]
        # torch.manual_seed(index if not self.test else self.max - index)
        # if self.buffer.get(f'init_cond_{index}') is None:
        # w0 = self.sampler.sample()
        # state, _ = self.navier_stokes_2d(w0, f=self.forcing_zero, visc=8e-4, T=30.0,
        # delta_t=self.dt, record_steps=20)
        # init_cond = state[:, :, -1, 0]
        # self.buffer[f'init_cond_{index}'] = init_cond.numpy()
        # else:
        # init_cond = torch.from_numpy(self.buffer[f'init_cond_{index}'])

        # return init_cond

    def create_envs(self, mode="train"):
        envs = []
        for i in range(self.n_env):
            envs.append(self.create_env(mode))
        return envs

    def create_env(self, mode="train"):
        env = dict()
        value_total = {"alpha": (-0.6, 0.6), "alpha1": (0, 3), "alpha2": (0, 3), "beta": (-0.6, 0.6), "beta1": (0, 3),
                       "beta2": (0, 3), "visc": (0, 90000)}
        value_test = {"alpha": (-0.2, 0.2), "alpha1": (1, 2), "alpha2": (1, 2), "beta": (-0.2, 0.2), "beta1": (1, 2),
                      "beta2": (1, 2), "visc": (30000, 60000)}
        if mode in ["train"]:
            env["alpha"] = random.uniform(*value_total["alpha"])
            env["alpha1"] = random.uniform(*value_total["alpha1"])
            env["alpha2"] = random.uniform(*value_total["alpha2"])
            env["beta"] = random.uniform(*value_total["beta"])
            env["beta1"] = random.uniform(*value_total["beta1"])
            env["beta2"] = random.uniform(*value_total["beta2"])
            env["visc"] = random.uniform(*value_total["visc"])

            flag = True
            for param in value_test.keys():
                if not (value_test[param][0] <= env[param] <= value_test[param][1]):
                    flag = False
                    break

            if flag:
                k = random.choices(list(env.keys()))[0]
                p = random.uniform(*value_total[k])
                while (value_test[param][0] <= p <= value_test[param][1]):
                    p = random.uniform(*value_total[k])
                env[k] = p

        else:
            env["alpha"] = random.uniform(*value_test["alpha"])
            env["alpha1"] = random.uniform(*value_test["alpha1"])
            env["alpha2"] = random.uniform(*value_test["alpha2"])
            env["beta"] = random.uniform(*value_test["beta"])
            env["beta1"] = random.uniform(*value_test["beta1"])
            env["beta2"] = random.uniform(*value_test["beta2"])
            env["visc"] = random.uniform(*value_test["visc"])
        return env

    # def create_env(self, mode="train"):
    # env = dict()
    # if mode in ["train"]:
    # env["alpha"] = random.uniform(0, 0.3)
    # env["beta"] = random.uniform(0, 0.3)
    # env["visc"] = random.uniform(0, 0.3)

    # params = list(env.values())
    # if all(((p >= 0.1 and p < 0.2) for p in params)):
    # k = random.choices(list(env.keys()))
    # p = random.uniform(0, 0.3)
    # while(0.1 <= p < 0.2):
    # p = random.uniform(0, 0.3)
    # env[k[0]] = p
    # else:
    # env["alpha"] = random.uniform(0.1, 0.2)
    # env["beta"] = random.uniform(0.1, 0.2)
    # env["visc"] = random.uniform(0.1, 0.2)
    # return env

    def generate_data(self):
        meta_dict = dict()
        meta_dict["envs_train"] = self.envs_train
        meta_dict["envs_test"] = self.envs_test
        with open(osp.join(self.file_path, "meta_data.pkl"), "wb") as f:
            pickle.dump(meta_dict, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env_cond)):
            env = self.envs_test[i]
            for j in tqdm(range(self.n_data_cond_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                states = states.permute(2, 0, 1, 3)
                inff, sup, states = self.normalization(states)
                times = times.unsqueeze(1)
                points = np.arange(times.shape[0])
                size = int(self.ratio * times.shape[0])
                if size < 1:
                    size = 1
                initial_loc = np.array([0])
                size -= 1
                points = points[1:]
                locations = np.random.choice(points, size=size, replace=False)
                locations = np.concatenate([initial_loc, locations])
                locations = np.sort(locations)
                times = times[locations, :]
                states = states[locations, :]
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                traj_dict["min"] = inff
                traj_dict["max"] = sup
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_cond.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env_cond)):
            env = self.envs_test[i]
            for j in tqdm(range(self.n_data_pred_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                states = states.permute(2, 0, 1, 3)
                inff, sup, states = self.normalization(states)
                times = times.unsqueeze(1)
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                traj_dict["min"] = inff
                traj_dict["max"] = sup
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_test.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env)):
            env = self.envs_train[i]
            for j in tqdm(range(self.n_data_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                states = states.permute(2, 0, 1, 3)
                inff, sup, states = self.normalization(states)
                times = times.unsqueeze(1)
                points = np.arange(times.shape[0])
                size = int(self.ratio * times.shape[0])
                if size < 1:
                    size = 1
                initial_loc = np.array([0])
                size -= 1
                points = points[1:]
                locations = np.random.choice(points, size=size, replace=False)
                locations = np.concatenate([initial_loc, locations])
                locations = np.sort(locations)
                times = times[locations, :]
                states = states[locations, :]
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                traj_dict["min"] = inff
                traj_dict["max"] = sup
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_train.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

    def generate_ts(self, env):
        w0 = self._get_init_cond()
        # env = self.envs[index]
        # print(env)
        times = np.arange(0., self.t_horizon, self.dt)
        # print(partial(self._f, env=env))
        states, times = self.navier_stokes_2d(w0, env,
                                              T=self.t_horizon, delta_t=self.dt, record_steps=self.n)
        # states = torch.from_numpy(y.y).float()
        # states = torch.transpose(states, 0, 1)
        # times = torch.from_numpy(times).float()
        # times = times.unsqueeze(-1)
        return times, env, states

    def is_legal(self, times, states):
        return True

    def normalization(self, states):
        inff = states.min()
        sup = states.max()
        states = (states - inff) / (sup - inff)
        return inff, sup, states


class GSGenerator(object):
    def __init__(self, name, n_env, n_data_per_env, t_horizon, dt, method="RK45", n_env_cond=None,
                 n_data_cond_per_env=None, n_data_pred_per_env=None, size=32, n_block=3, dx=1., random_influence=0.2,
                 ratio=0.5):
        tgt_path = osp.join(os.getcwd(), f"datasets/{name}")
        self.file_path = osp.join(tgt_path, f"ratio{ratio}")
        self.n_env = n_env
        self.n_data_per_env = n_data_per_env
        self.n_env_cond = n_env_cond
        self.n_data_cond_per_env = n_data_cond_per_env
        self.n_data_pred_per_env = n_data_pred_per_env
        self.time_horizon = t_horizon
        self.method = method
        self.dx = dx
        self.dt_eval = dt
        self.ratio = ratio
        self.envs_train = self.create_envs(mode="train")
        self.envs_test = self.create_envs(mode="test")
        self.size = size
        self.n_block = n_block

    def _laplacian2D(self, a):
        # a_nn | a_nz | a_np
        # a_zn | a    | a_zp
        # a_pn | a_pz | a_pp
        a_zz = a

        a_nz = np.roll(a_zz, (+1, 0), (0, 1))
        a_pz = np.roll(a_zz, (-1, 0), (0, 1))
        a_zn = np.roll(a_zz, (0, +1), (0, 1))
        a_zp = np.roll(a_zz, (0, -1), (0, 1))

        a_nn = np.roll(a_zz, (+1, +1), (0, 1))
        a_np = np.roll(a_zz, (+1, -1), (0, 1))
        a_pn = np.roll(a_zz, (-1, +1), (0, 1))
        a_pp = np.roll(a_zz, (-1, -1), (0, 1))

        return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (self.dx ** 2)

    def _vec_to_mat(self, vec_uv):
        UV = np.split(vec_uv, 2)
        U = np.reshape(UV[0], (self.size, self.size))
        V = np.reshape(UV[1], (self.size, self.size))
        return U, V

    def _mat_to_vec(self, mat_U, mat_V):
        dudt = np.reshape(mat_U, self.size * self.size)
        dvdt = np.reshape(mat_V, self.size * self.size)
        return np.concatenate((dudt, dvdt))

    def _f(self, t, uv, env=0):
        U, V = self._vec_to_mat(uv)
        deltaU = self._laplacian2D(U)
        deltaV = self._laplacian2D(V)
        dUdt = (env['r_u'] * deltaU - U * (V ** 2) + env['f'] * (1. - U))
        dVdt = (env['r_v'] * deltaV + U * (V ** 2) - (env['f'] + env['k']) * V)
        duvdt = self._mat_to_vec(dUdt, dVdt)
        return duvdt

    def _get_init_cond(self):
        size = (self.size, self.size)
        U = 0.95 * np.ones(size)
        V = 0.05 * np.ones(size)
        for _ in range(self.n_block):
            r = int(self.size / 10)
            N2 = np.random.randint(low=0, high=self.size - r, size=2)
            U[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 0.
            V[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 1.
        return U, V

    def create_envs(self, mode="train"):
        envs = []
        for i in range(self.n_env):
            envs.append(self.create_env(mode))
        return envs

    def create_env(self, mode="train"):
        env = dict()
        value_total = {"f": (2.25e-2, 4.35e-2), "k": (5.6e-2, 6.4e-2), "r_u": (0.0, 0.3), "r_v": (0.0, 0.3)}
        value_test = {"f": (3e-2, 3.9e-2), "k": (5.8e-2, 6.2e-2), "r_u": (0.1, 0.2), "r_v": (0.1, 0.2)}
        if mode in ["train"]:
            env["f"] = random.uniform(*value_total["f"])
            env["k"] = random.uniform(*value_total["k"])
            env["r_u"] = random.uniform(*value_total["r_u"])
            env["r_v"] = random.uniform(*value_total["r_v"])

            flag = True
            for param in value_test.keys():
                if not (value_test[param][0] <= env[param] <= value_test[param][1]):
                    flag = False
                    break

            if flag:
                k = random.choices(list(env.keys()))[0]
                p = random.uniform(*value_total[k])
                while (value_test[param][0] <= p <= value_test[param][1]):
                    p = random.uniform(*value_total[k])
                env[k] = p

        else:
            env["f"] = random.uniform(*value_total["f"])
            env["k"] = random.uniform(*value_total["k"])
            env["r_u"] = random.uniform(*value_total["r_u"])
            env["r_v"] = random.uniform(*value_total["r_v"])
        return env

    def generate_data(self):
        meta_dict = dict()
        meta_dict["envs_train"] = self.envs_train
        meta_dict["envs_test"] = self.envs_test
        with open(osp.join(self.file_path, "meta_data.pkl"), "wb") as f:
            pickle.dump(meta_dict, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env_cond)):
            env = self.envs_test[i]
            for j in tqdm(range(self.n_data_cond_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                # print(states)
                # print(states["y"].shape)
                # states = torch.FloatTensor(states["y"]).permute(1, 0)
                # print(states.shape)
                # sys.exit()
                inff, sup, states = self.normalization(states)
                # print(states.shape)
                # times = times.unsqueeze(1)
                points = np.arange(times.shape[0])
                size = int(self.ratio * times.shape[0])
                if size < 1:
                    size = 1
                initial_loc = np.array([0])
                size -= 1
                points = points[1:]
                locations = np.random.choice(points, size=size, replace=False)
                locations = np.concatenate([initial_loc, locations])
                locations = np.sort(locations)
                times = times[locations, :]
                states = states[locations, :]
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                traj_dict["min"] = inff
                traj_dict["max"] = sup
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_cond.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env_cond)):
            env = self.envs_test[i]
            for j in tqdm(range(self.n_data_pred_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                # states = states.permute(2, 0, 1, 3)
                inff, sup, states = self.normalization(states)
                # times = times.unsqueeze(1)
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                traj_dict["min"] = inff
                traj_dict["max"] = sup
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_test.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

        trajs = []
        for i in tqdm(range(self.n_env)):
            env = self.envs_train[i]
            for j in tqdm(range(self.n_data_per_env)):
                traj_dict = dict()
                while (True):
                    times, params, states = self.generate_ts(env)
                    if self.is_legal(times, states):
                        break
                # states = states.permute(2, 0, 1, 3)
                inff, sup, states = self.normalization(states)
                # times = times.unsqueeze(1)
                points = np.arange(times.shape[0])
                size = int(self.ratio * times.shape[0])
                if size < 1:
                    size = 1
                initial_loc = np.array([0])
                size -= 1
                points = points[1:]
                locations = np.random.choice(points, size=size, replace=False)
                locations = np.concatenate([initial_loc, locations])
                locations = np.sort(locations)
                times = times[locations, :]
                states = states[locations, :]
                traj_dict["x"] = times
                traj_dict["y"] = states
                traj_dict["index"] = torch.IntTensor([i])
                traj_dict["min"] = inff
                traj_dict["max"] = sup
                trajs.append(traj_dict)
        with open(osp.join(self.file_path, "trajs_train.pkl"), "wb") as f:
            pickle.dump(trajs, f)
        f.close()

    def generate_ts(self, env):
        uv_0 = self._mat_to_vec(*self._get_init_cond())
        times = np.arange(0., self.time_horizon, self.dt_eval)
        y = solve_ivp(partial(self._f, env=env), (0., self.time_horizon), y0=uv_0, method=self.method,
                      t_eval=np.arange(0., self.time_horizon, self.dt_eval))
        states = torch.from_numpy(y.y).float()
        states = torch.transpose(states, 0, 1)
        times = torch.from_numpy(times).float()
        times = times.unsqueeze(-1)
        return times, env, states

    def is_legal(self, times, states):
        return True

    def normalization(self, states):
        inff = states.min()
        sup = states.max()
        states = (states - inff) / (sup - inff)
        return inff, sup, states


if __name__ == "__main__":
    random.seed(2813)
    np.random.seed(2813)
    torch.manual_seed(2813)
    torch.cuda.manual_seed_all(2813)
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="LV")
    parser.add_argument("--ratio", type=float, default=0.5)
    args = parser.parse_args()
    # if args.name == "LV":
    generator = LVGenerator(name="LV", n_env=9, n_data_per_env=10, t_horizon=10, dt=0.5, method="RK45", n_env_cond=9, n_data_cond_per_env=10, n_data_pred_per_env=32, ratio=args.ratio)
    # tgt_path = osp.join(os.getcwd(), "datasets/NS")
    #generator = NSGenerator(name="NS", n_env=5, n_data_per_env=10, t_horizon=50, dt_eval=5, method="RK45", n_env_cond=5,
                            #n_data_cond_per_env=10, n_data_pred_per_env=10, ratio=args.ratio)
    # generator = GSGenerator(name="GS", n_env=9, n_data_per_env=10, t_horizon=10, dt=0.5, method="RK45", n_env_cond=9, n_data_cond_per_env=10, n_data_pred_per_env=32, ratio=args.ratio)
    # generator = GOGenerator(name="GO", n_env=9, n_data_per_env=10, t_horizon=10, dt=0.5, method="RK45", n_env_cond=9, n_data_cond_per_env=10, n_data_pred_per_env=32, ratio=args.ratio)
    generator.generate_data()