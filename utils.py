import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import init
from torch import nn


def Linear(args, output_size, bias, bias_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias(default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.data.size(1) for a in args]
    for shape in shapes:
        total_arg_size += shape

    # Now the computation.
    weights = nn.Parameter(torch.FloatTensor(total_arg_size, output_size))
    init.xavier_uniform_(weights)
    # weights = Variable(torch.zeros(total_arg_size, output_size))
    if len(args) == 1:
        res = torch.matmul(args[0], weights)  # torch.matmul是tensor的乘法
    else:
        x = torch.cat(args, 1)
        res = torch.matmul(torch.cat(args, 1), weights)
        # torch.cat(args, 1)横着拼
    if not bias:
        return res

    if bias_initializer is None:
        biases = Variable(torch.zeros(output_size))
    return torch.add(res, biases)


def basic_hyperparams():
    return {
        # model parameters
        'learning_rate': 1e-3,
        'lambda_l2_reg': 1e-3,
        'gc_rate': 2.5,  # to avoid gradient exploding
        'dropout_rate': 0.3,
        'n_stacked_layers': 2,
        's_attn_flag': 2,
        'ext_flag': True,

        # encoder parameter
        'n_sensors': 35,
        'n_input_encoder': 4,
        'n_steps_encoder': 240,  # time steps
        'n_hidden_encoder': 64,  # size of hidden units

        # decoder parameter
        'n_steps_decoder': 72,
        'n_hidden_decoder': 64,
        'n_output_decoder': 1  # size of the decoder output
    }


def shuffle_data(training_data):
    """ shuffle data"""
    shuffle_index = np.random.permutation(training_data[0].shape[0])
    new_training_data = []
    for inp in training_data:
        new_training_data.append(inp[shuffle_index])
    return new_training_data


def get_batch_feed_dict(k, batch_size, training_data, train_labels_data):
    batch_train_inp = training_data[k:k + batch_size]
    batch_label_inp = train_labels_data[k:k + batch_size]
    feed_dict = (batch_train_inp, batch_label_inp)
    return feed_dict


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))* (560 - 2) + 2


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)* (560 - 2) + 2

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)* (560 - 2) + 2