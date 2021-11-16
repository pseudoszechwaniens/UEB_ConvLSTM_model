# coding: utf-8

from mxnet import nd
import numpy as np
import copy

def validate(data_iter, state, net, loss, batch_size, l_sum, n, i, epoch, epochs):
    """
    Validate loss of a model on the given data set.
    """
    for x, y in data_iter:
        assert len(x) == batch_size
        out, _ = net(x, state)
        l = loss(y, out)
        l_sum += l.sum().asscalar()
        n += y.size
    return l_sum, n
	
	
def NSE(y_pred, y_true):
    """
    Calculate the Nash-Sutcliffe efficiency coefficient.
    """
    return 1 - sum((y_pred - y_true) ** 2) / sum((y_true - y_true.mean()) ** 2)

def PBIAS(y_pred, y_true):
    """
    Calculate the percent bias.
    """
    return 100 * sum(y_true - y_pred) / sum(y_true)