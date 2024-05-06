import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset
from functools import partial
import math
import shelve


class ODEDataset(Dataset):
    def __init__(self, n_data_per_env, params, t_horizon, dt, random_influence=0.2, method='RK45',
                 group="train", rdn_gen=1.):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.t_horizon = float(t_horizon)
        self.dt = dt
        self.random_influence = random_influence
        self.params_eq = params
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.method = method
        self.rdn_gen = rdn_gen

    def _f(self, t, x, env=0):
        raise NotImplemented

    def _get_init_cond(self, index):
        raise NotImplemented

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.t_horizon, self.dt).float()
        out = {'t': t, 'env': env}
        if self.buffer.get(index) is None:
            y0 = self._get_init_cond(env_index)
            y = solve_ivp(partial(self._f, env=env), (0., self.t_horizon), y0=y0, method=self.method,
                          t_eval=np.arange(0., self.t_horizon, self.dt))
            y = torch.from_numpy(y.y).float()
            out['state'] = y
            self.buffer[index] = y.numpy()
        else:
            out['state'] = torch.from_numpy(self.buffer[index])

        out['index'] = index
        out['param'] = torch.tensor(list(self.params_eq[env].values()))
        return out

    def __len__(self):
        return self.len


class LotkaVolterraDataset(ODEDataset):
    def _f(self, t, x, env=0):
        alpha = self.params_eq[env]['alpha']
        beta = self.params_eq[env]['beta']
        gamma = self.params_eq[env]['gamma']
        delta = self.params_eq[env]['delta']
        d = np.zeros(2)
        d[0] = alpha * x[0] - beta * x[0] * x[1]
        d[1] = delta * x[0] * x[1] - gamma * x[1]
        return d

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max - index)
        return np.random.random(2) + self.rdn_gen