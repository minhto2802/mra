"""Brain MRA"""
import glob
import pickle
import pylab as plt
from mxnet.gluon import nn, Trainer, loss
from mxnet import autograd, nd, initializer, io, gpu
import numpy as np
import os
import argparse
import time
from mxboard import SummaryWriter


def parse_args():
    """Get commandline parameters"""
    parser = argparse.ArgumentParser(description='Run CNN training')
    parser.add_argument('--run_id', type=int, default=25)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--file_idx', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_unit', type=int, default=3)
    parser.add_argument('--num_stage', type=int, default=8)
    parser.add_argument('--channels', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--resumed_epoch', type=int, default=99)
    parser.add_argument('--resumed_epoch', type=int, default=-1)
    parser.add_argument('--train_amount', type=int, default=40)
    parser.add_argument('--val_amount', type=int, default=17)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--dir_in', type=str, default=r"C:\Workspace\KU AIS Patient Anonymized/")
    parser.add_argument('--dir_out', type=str, default=r"F:\BACKUPS\MRA/")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    # parser.add_argument('--validation_only', type=int, default=1)
    parser.add_argument('--validation_only', type=int, default=-1)
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
                                label=np.zeros((opts.train_amount, )), batch_size=opts.batch_size)
    val_iter = io.NDArrayIter(data=np.arange(opts.val_amount),
                              label=np.zeros((opts.val_amount, )), batch_size=opts.batch_size)
    return train_iter, val_iter


def set_context(opts):
    """Setup context"""
    return gpu(opts.gpu_id)


def set_trainer(opts, net):
    """Setup training policy"""
    return Trainer(net.collect_params(), optimizer=opts.optimizer,
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


def conv_factory(k=3, channels=8, bn=False):
    """A convenient conv implementation"""
    body = nn.HybridSequential()
    if bn:
        body.add((nn.BatchNorm()))
    p = int((k - 1)/2)
    body.add((nn.Conv3D(kernel_size=k, padding=p, strides=1,
                        channels=channels)))
    return body


class ResBlock(nn.HybridBlock):
    """Residual Block"""
    def __init__(self, k=3, channels=8):
        super(ResBlock, self).__init__()
        self.body = nn.HybridSequential()
        self.body.add((conv_factory(k, channels=channels)))
        self.body.add((conv_factory(k, channels=channels)))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        return x + self.body(x)


class Network(nn.HybridBlock):
    """Construct the network"""
    def __init__(self, opts):
        super(Network, self).__init__()
        self.body = nn.HybridSequential()
        self.body.add(conv_factory(k=opts.kernel_size, channels=opts.channels, bn=False))
        for stage in range(opts.num_stage):
            for unit in range(opts.num_unit):
                self.body.add(ResBlock(channels=opts.channels))
            if stage == opts.num_stage - 1:
                channels = 1
                self.body.add(conv_factory(k=1, channels=channels))

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Forward"""
        return self.body(x)


def validate(opts, val_list, file, val_iter, sw=None, num_plot=None):
    """Validation"""
    if num_plot is None:
        num_plot = val_list.__len__()
    loss_cumulative = []
    pred_list = []
    lab_list = []
    val_iter.reset()
    count = 0  # count number of pred and lab retained
    for (i, batch) in enumerate(val_iter):
        idx = batch.data[0].asnumpy()
        data = nd.zeros((args.batch_size, 60, 20, 240, 240), ctx=ctx)
        label = nd.zeros((args.batch_size, 20, 240, 240), ctx=ctx)
        for (j, jj) in zip(idx, range(opts.batch_size)):
            data[jj] = np.load("%sIMG.npy" % val_list[int(j)])
            label[jj] = np.load("%s%s.npy" % (val_list[int(j)], file))

        pred = net(data)
        L = loss(pred, label)
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


def validate_only(opts):
    """Run prediction on the validation set (without running training)"""
    _, val_iter = set_iter(opts)
    _, val_list = get_subject_list(opts)
    file, dir_checkpoints = prepare_dir(opts)
    loss_val, preds, labs = validate(opts, val_list, file, val_iter, sw)
    for k in range(preds.shape[0]):
        im = integrate_pred_lab(integrate_slices(norm01(preds[k])), integrate_slices(norm01(labs[k])))
    print('Validation loss: {:.2f}, at epoch {:03d}'.format(loss_val, opts.resumed_epoch))
    np.save("%sval_pred_Epoch%03d" % (opts.dir_out, opts.resumed_epoch), preds.asnumpy())
    np.save("%sval_lab_Epoch%03d" % (opts.dir_out, opts.resumed_epoch), labs.asnumpy())


def train(opts, net, sw):
    """Training"""
    first_val = False
    train_list, val_list = get_subject_list(opts)
    trainer = set_trainer(opts, net)
    file, dir_checkpoints = prepare_dir(opts)
    train_iter, val_iter = set_iter(opts)
    tic = time.time()
    lowest_loss = np.Inf
    for epoch in range(opts.epochs):
        etic = time.time()
        btic = time.time()
        train_iter.reset()
        loss_cumulative = []
        for (i, batch) in enumerate(train_iter):
            idx = batch.data[0].asnumpy()
            data = nd.zeros((args.batch_size, 60, 20, 240, 240), ctx=ctx)
            label = nd.zeros((args.batch_size, 20, 240, 240), ctx=ctx)
            for (j, jj) in zip(idx, range(opts.batch_size)):
                data[jj] = np.load("%sIMG.npy" % train_list[int(j)])
                label[jj] = np.load("%s%s.npy" % (train_list[int(j)], file))
            with autograd.record():
                pred = net(data)
                L = loss(pred, label)
                L.backward()
            loss_cumulative.append((L.expand_dims(1)))
            trainer.step(opts.batch_size)
            if (i + 1) % opts.log_interval == 0:
                print('[Epoch {}/{}] [Batch {}] Loss: {:.3f}, {:.2f} Samples / sec'.
                      format(epoch, opts.epochs, i, L.asscalar(), opts.log_interval * opts.batch_size / (time.time() - btic)))
            btic = time.time()
        loss_train = nd.concat(*loss_cumulative, dim=0).mean().asscalar()

        sw.add_scalar('loss', ('training', loss_train), global_step=epoch)
        if (epoch + 1) % opts.val_interval == 0:
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
                if loss_val == lowest_loss:
                    net.save_params("%sBest_epoch.params" % dir_checkpoints)
            first_val = True

            # Log training predictions
            im = integrate_pred_lab(integrate_slices(norm01(nd.squeeze(pred[-1]))),
                                    integrate_slices(norm01(nd.squeeze(label[-1]))))
            sw.add_image('train_predictions', norm01(im), global_step=epoch)

        sw_cmd = "cd '%s' \ntensorboard --logdir=./logs --host=127.0.0.1 --port=8888" % opts.dir_out
        print('Copy paste this to view Tensorboard:\n%s' % sw_cmd)

        print('Training loss: {:.2f}'.format(loss_train))

        if first_val and ((epoch + 1) % opts.checkpoint_interval == 0):
            net.save_params("%sEpoch%03d.params" % (dir_checkpoints, epoch))

            print('Validation loss: {:.2f}, '
                  '\nBest Validation Loss: {:.2f} at epoch {:03d}'
                  '\nEpoch time: {:.2f} secs'.
                  format(loss_val, lowest_loss, best_epoch, time.time() - etic))

    print('Total training time: {:.2f} secs'.format(time.time() - tic))


if __name__ == "__main__":
    args = parse_args()
    ctx = set_context(args)
    net = Network(args)
    x = nd.random_normal(0.02, 0.2, shape=(args.batch_size, 60, 20, 240, 240), ctx=ctx)

    if args.resumed_epoch < 0:
        net.collect_params().initialize(initializer.Xavier(magnitude=2), ctx=ctx)
    else:
        net.load_params('%s/checkpoints/Epoch%03d.params' % (args.dir_out, args.resumed_epoch), ctx=ctx)
    net(x)
    net.hybridize()

    loss = SmoothL1Loss()
    sw = SummaryWriter(logdir="%slogs" % args.dir_out, flush_secs=5,
                       filename_suffix='mra', verbose=False)
    if args.validation_only > 0:
        validate_only(args)
    else:
        train(args, net, sw)



