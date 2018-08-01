"""Brain MRA"""
import glob
import pickle
import pylab as plt
from mxnet.gluon import nn, Trainer, loss, Block, rnn, utils
from mxnet import autograd, nd, initializer, io, gpu, cpu, init
import numpy as np
import os
import argparse
import time
from mxboard import SummaryWriter


def parse_args():
    """Get commandline parameters"""
    parser = argparse.ArgumentParser(description='Run CNN training')
    parser.add_argument('--run_id', type=int, default=19)
    parser.add_argument('--batch_size0', type=int, default=1)
    parser.add_argument('--batch_size1', type=int, default=64)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_unit', type=int, default=3)
    parser.add_argument('--num_stage', type=int, default=8)
    parser.add_argument('--channels', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--file_idx', type=int, default=0)
    parser.add_argument('--train_amount', type=int, default=40)
    parser.add_argument('--val_amount', type=int, default=17)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--dir_in', type=str, default=r"C:\Workspace\KU AIS Patient Anonymized/")
    parser.add_argument('--dir_out', type=str, default=r"F:\BACKUPS\MRA/")
    parser.add_argument('--model', type=str, default='rnn_relu')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--pre_run', type=bool, default=False)
    args = parser.parse_args()
    args.dir_out = "%srun%03d/" % (args.dir_out, args.run_id)
    return args


def norm01(x):
    """Normalize all values to range 0 - 1"""
    return (x - x.min()) / (x.max() - x.min())


def prepare_dir(opts):
    """Assigning and creating directories"""
    dir_checkpoints = "%s/checkpoints/" % opts.dir_out
    for d in (opts.dir_out, dir_checkpoints):
        if not os.path.exists(d):
            os.mkdir(d)
    files = ['ColArt', 'ColCap', 'ColDel', 'ColEVen', 'ColLVen', 'IMG']
    file = files[opts.file_idx]
    return file, dir_checkpoints


def get_subject_list(opts):
    """Get the list of subjects"""
    with open("%ssubject_list.txt" % opts.dir_in, "rb") as fp:   # Unpickling
        subject_list = pickle.load(fp)
    train_list = subject_list[:opts.train_amount]
    val_list = subject_list[opts.train_amount: opts.train_amount + opts.val_amount]
    return train_list, val_list


def set_iter(opts):
    """data iterator"""
    train_iter = io.NDArrayIter(data=np.arange(opts.train_amount), shuffle=True,
                                label=np.zeros((opts.train_amount, )), batch_size=opts.batch_size0)
    val_iter = io.NDArrayIter(data=np.arange(opts.val_amount),
                              label=np.zeros((opts.val_amount, )), batch_size=opts.batch_size0)
    return train_iter, val_iter


def set_context(opts):
    """Setup context"""
    context = gpu(opts.gpu_id) if opts.gpu_id > -1 else cpu()
    return context


def set_trainer(opts, model):
    """Setup training policy"""
    return Trainer(model.collect_params(), optimizer=opts.optimizer,
                   optimizer_params={'learning_rate': opts.base_lr,
                                     'wd': opts.wd})


class SmoothL1Loss(loss.Loss):
    """Smooth L1 Loss"""
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask=None):
        """Forward"""
        if mask is None:
            mask = nd.ones(label.shape).as_in_context(label.context)
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.mean(loss, self._batch_axis, exclude=True)


def _reshape(arr, batch_size=32):
    if len(arr.shape) == 3:
        arr = arr.reshape((arr.shape[0], -1))
        num_batches = arr.shape[-1] // batch_size
        return arr.reshape((arr.shape[0], batch_size, num_batches)).transpose((2, 1, 0))
    else:
        arr = arr.reshape((arr.shape[0], arr.shape[1], -1))
        num_batches = arr.shape[-1] // batch_size
        return arr.reshape((arr.shape[:-1] + (batch_size, num_batches))).transpose((0, 3, 2, 1)).swapaxes(0, 1)


def _shape_recover(arr, shape, batch_size=128):
    if len(arr.shape) == 3:
        arr = arr.transpose((2, 1, 0)).reshape((shape[0], shape[-2] * shape[-1]))
        return arr.reshape(shape)
    else:
        arr = arr.swapaxes(0, 1).transpose((0, 3, 2, 1)).reshape((shape[0], shape[1], shape[-2] * shape[-1]))
        return arr.reshape((shape[0], shape[1], shape[-2], shape[-1]))


class RNNModel(Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, num_hidden, num_layers, num_slices=20, dropout=0.3, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru" % mode)
            self.decoder = nn.Dense(num_slices, in_units=num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        outputs = []
        for X in inputs:  # (60 x 64 x 20):
            output, hidden = self.rnn(X.expand_dims(0), hidden)
            output = self.drop(nd.squeeze(output))
        decoded = self.decoder(output)
        output_final = decoded.expand_dims(0)
        # outputs.append(decoded.expand_dims(0))
        # output_final = nd.concat(*outputs, dim=0).mean(axis=0)
        # return output_final, hidden
        return output_final, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


def validate(opts, val_list, file, val_iter, sw, model, num_plot=4):
    """Validation"""
    loss_cumulative = []
    pred_list = []
    lab_list = []
    val_iter.reset()
    count = 0  # count number of pred and lab retained
    for (i, batch) in enumerate(val_iter):
        idx = batch.data[0].asnumpy()
        data = nd.zeros((opts.batch_size0, 60, 20, 240, 240), ctx=ctx)
        label = nd.zeros((opts.batch_size0, 20, 240, 240), ctx=ctx)
        for (j, jj) in zip(idx, range(opts.batch_size0)):
            data[jj] = np.load("%sIMG.npy" % val_list[int(j)])
            label[jj] = np.load("%s%s.npy" % (val_list[int(j)], file))
        with autograd.record():
            pred = model(data)
            L = loss(pred, label)
            L.backward()
        loss_cumulative.append((L.expand_dims(1)))
        if count < num_plot:
            pred_list.append(pred)
            lab_list.append(label)
            count += 1
    return nd.concat(*loss_cumulative, dim=0).mean().asscalar(), \
           nd.squeeze(nd.concat(*pred_list, dim=0)), \
           nd.squeeze(nd.concat(*lab_list, dim=0))


def integrate_slices(im3d):
    """Integrating different slices of 1 3D image into one 2D image"""
    for sl in range(im3d.shape[0]):
        if sl == 0:
            im = im3d[sl]
        else:
            im = nd.concat(im, im3d[sl], dim=-1)
    return im


def integrate_pred_lab(pred, lab):
    """Integrating prediction and label into two rows of one image"""
    return nd.concat(pred, lab, dim=-2)


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def train(opts, model, sw):
    """Training"""
    train_list, val_list = get_subject_list(opts)
    trainer = set_trainer(opts, model)
    file, dir_checkpoints = prepare_dir(opts)
    train_iter, val_iter = set_iter(opts)
    tic = time.time()
    lowest_loss = np.Inf
    global_step = 0
    for epoch in range(opts.epochs):
        sw_cmd = "cd '%s' \ntensorboard --logdir=./logs --host=127.0.0.1 --port=8888" % opts.dir_out
        print('Copy paste this to view Tensorboard:\n%s' % sw_cmd)
        etic = time.time()
        btic = time.time()
        train_iter.reset()
        loss_cumulative = []
        for (i, batch) in enumerate(train_iter):
            idx = batch.data[0].asnumpy()

            data2d = np.load("%sIMG.npy" % train_list[int(idx)])
            label2d = np.load("%s%s.npy" % (train_list[int(idx)], file))

            data = nd.array(_reshape(data2d[..., 80:112, 80:112], batch_size=opts.batch_size1), ctx=ctx)  # [..., 80:112, 80:112]
            label = nd.array(_reshape(label2d[..., 80:112, 80:112], batch_size=opts.batch_size1), ctx=ctx)
            preds = nd.zeros(label.shape)
            # hidden = detach(hidden)

            for i_batch in range(data.shape[0]):
                hidden = model.begin_state(func=nd.zeros, batch_size=opts.batch_size1, ctx=ctx)
                with autograd.record():
                    # pred = net(data)
                    pred, hidden = model(data[i_batch], hidden)
                    L = loss(pred, label[i_batch])
                    R = np.corrcoef(pred.asnumpy().flatten(),
                                      label[i_batch].asnumpy().flatten())[0, 1]
                    print(R)
                    L.backward()

                # grads = [i.grad(ctx) for i in model.collect_params().values()]
                # utils.clip_global_norm(grads, opts.clip * opts.batch_size1)

                trainer.step(opts.batch_size1)

                loss_cumulative.append((L.expand_dims(1)))
                preds[i_batch] = pred
                if (i_batch + 1) == preds.shape[0]:
                    # if (i_batch + 1) == 1:
                    u0 = _shape_recover(label, (1, 20, 32, 32))
                    u1 = _shape_recover(preds, (1, 20, 32, 32)).as_in_context(ctx)
                    v0 = u0[0, 0]
                    v1 = u1[0, 0]
                    for kk in range(1, u0.shape[1]):
                        v0 = nd.concat(v0, u0[0, kk], dim=-1)
                        v1 = nd.concat(v1, u1[0, kk], dim=-1)
                    vv = nd.concat(v0, v1, dim=-2)
                    sw.add_image('label_prediction', norm01(vv), global_step=global_step)
                    global_step += 1
                plt.show()
                if (i + 1) % opts.log_interval == 0:
                    print('[Epoch {}/{}] [Batch {}-{}] Loss: {:.3f}, {:.2f} Samples / sec'.
                          format(epoch, opts.epochs, i, i_batch, L.mean().asscalar(),
                                 opts.log_interval * opts.batch_size1 / (time.time() - btic)))
                btic = time.time()

        loss_train = nd.concat(*loss_cumulative, dim=0).mean().asscalar()

        sw.add_scalar('loss', ('training', loss_train), global_step=epoch)
        loss_val = lowest_loss
        if opts.val_interval & (epoch + 1) % opts.val_interval == 0:
            loss_val, preds, labs = validate(opts, val_list, file, val_iter, sw)
            sw.add_scalar('loss', ('validation', loss_val), global_step=epoch)
            # Log validation predictions
            for k in range(preds.shape[0]):
                im = integrate_pred_lab(integrate_slices(norm01(preds[k])), integrate_slices(norm01(labs[k])))
                sw.add_image('val_predictions_%02d' % k, im, global_step=epoch)
            if loss_val <= lowest_loss:
                lowest_loss = loss_val
                np.save("%sval_pred_best_epoch" % opts.dir_out, preds.asnumpy())
                np.save("%sval_lab_best_epoch" % opts.dir_out, labs.asnumpy())
                best_epoch = epoch

            # Log training predictions
            im = integrate_pred_lab(integrate_slices(norm01(nd.squeeze(pred[-1]))),
                                    integrate_slices(norm01(nd.squeeze(label[-1]))))
            sw.add_image('train_predictions', norm01(im), global_step=epoch)

        sw_cmd = "cd '%s' \ntensorboard --logdir=./logs --host=127.0.0.1 --port=8888" % opts.dir_out
        print('Copy paste this to view Tensorboard:\n%s' % sw_cmd)

        if (epoch + 1) % opts.checkpoint_interval == 0:
            model.save_params("%sEpoch%03d.params" % (dir_checkpoints, epoch))
        if loss_val == lowest_loss:
            model.save_params("%sBest_epoch.params" % dir_checkpoints)

        print('Training loss: {:.2f},  Validation loss: {:.2f}, '
              '\nBest Validation Loss: {:.2f} at epoch {:03d}'
              '\nEpoch time: {:.2f} secs'.
              format(loss_train, loss_val, lowest_loss, best_epoch, time.time() - etic))
    print('Total training time: {:.2f} secs'.format(time.time() - tic))


if __name__ == "__main__":
    args = parse_args()
    ctx = set_context(args)

    n_hidden = 100
    n_slices = 20

    model = RNNModel(args.model, n_hidden, args.num_layers, num_slices=n_slices)
    model.collect_params().initialize(init.Xavier(), ctx=ctx)

    if args.pre_run:
        hidden = model.begin_state(func=nd.zeros, batch_size=32, ctx=ctx)
        X = nd.random_normal(shape=(60, 32, 20))
        o, h = model(X, hidden)
        print(o.shape)
        print(h.__len__(), h[0].shape)

    loss = SmoothL1Loss()
    # loss = loss.L2Loss()
    sw = SummaryWriter(logdir="%slogs" % args.dir_out, flush_secs=5,
                       filename_suffix='mra', verbose=False)
    train(args, model, sw)



