"""Test RNN"""
import argparse
import pickle
import numpy as np
from mxnet.gluon import nn, Trainer, loss, rnn
from mxnet import autograd, nd, initializer, io, gpu


def parse_args():
    """Get commandline parameters"""
    parser = argparse.ArgumentParser(description='Run CNN training')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_unit', type=int, default=3)
    parser.add_argument('--num_stage', type=int, default=8)
    parser.add_argument('--channels', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--file_idx', type=int, default=0)
    parser.add_argument('--train_amount', type=int, default=40)
    parser.add_argument('--val_amount', type=int, default=17)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--dir_in', type=str, default=r"C:\Workspace\KU AIS Patient Anonymized/")
    parser.add_argument('--dir_out', type=str, default=r"F:\BACKUPS\MRA/")
    parser.add_argument('--run_id', type=int, default=9)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    args = parser.parse_args()
    args.dir_out = "%srun%03d/" % (args.dir_out, args.run_id)
    return args


def get_subject_list(opts):
    """Get the list of subjects"""
    with open("%ssubject_list.txt" % opts.dir_in, "rb") as fp:   # Unpickling
        subject_list = pickle.load(fp)
    train_list = subject_list[:opts.train_amount]
    val_list = subject_list[opts.train_amount: opts.train_amount + opts.val_amount]
    return train_list, val_list


def set_context(opts):
    """Setup context"""
    return gpu(opts.gpu_id)


class Network(nn.HybridBlock):
    """"Network"""
    def __init__(self):
        super(Network, self).__init__()
        # self.body = nn.

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""


def run(opts):
    """Run"""
    ctx = set_context(args)
    data = nd.zeros((args.batch_size, 60, 20, 240, 240), ctx=ctx)
    train_list, val_list = get_subject_list(opts)
    j = 0
    jj = 0
    data[jj] = np.load("%sIMG.npy" % val_list[int(j)])


if __name__ == "__main__":
    args = parse_args()
    # run(args)

    model = nn.Sequential()
    with model.name_scope():
        model.add(nn.Flatten())
        model.add(nn.Embedding(30, 10))
        model.add(rnn.LSTM(20))
        # model.add(nn.Dense(5, flatten=False))
    model.initialize()
    print(model(nd.ones((2, 3, 5))))