import numpy as np
from random import randint
import sys


# def set_requires_grad(module, tf=False):
#    module.requires_grad = tf
#    for param in module.parameters():
#        param.requires_grad = tf

# def condition_prediction_split(x, y, use_y0):
#    num_points = x.shape[1]
#    points = np.arange(num_points)
#    initial_loc = np.array([])
#    if use_y0:
#        points = points[1:]

#    size = randint(0, len(points))
#    locations = np.random.choice(points, size=size, replace=False)
#    locations = np.concatenate([initial_loc, locations])

#    x_context = x[:, locations[:num_co]]

def test_condition_split(x, y, use_y0):
    num_points = x.shape[1]
    points = np.arange(num_points)
    initial_loc = np.array([])
    if use_y0:
        points = points[1:]

    size = randint(0, len(points))
    locations = np.random.choice(points, size=size, replace=False)
    locations = np.concatenate([initial_loc, locations])
    x_condition = x[:, locations, :]
    y_condition = y[:, locations, :]
    return x_condition, y_condition, y[:, 0, :]


def context_target_split(x, y, num_context, num_extra_target, locations=None, use_y0=True):
    num_points = x.shape[1]

    if locations is None:
        points = np.arange(num_points)
        size = num_context + num_extra_target
        initial_loc = np.array([])
        if use_y0:
            points = points[1:]
            size -= 1
            initial_loc = np.array([0])

        locations = np.random.choice(points, size=size, replace=False)
        locations = np.concatenate([initial_loc, locations])
    locations = np.sort(locations)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target, y[:, 0, :]

# def condition_split(x, y):
#    num_points = x.shape[1]
#    points = np.ar
#    size = randint(0, len(points) - 1)