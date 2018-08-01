"""Plot prediction vs label for Professor"""
import numpy as np
import pylab as plt


def integrate_slices(im3d):
    """Integrating different slices of 1 3D image into one 2D image"""
    for sl in range(im3d.shape[0]):
        if sl == 0:
            im = im3d[sl]
        else:
            im = np.concatenate((im, im3d[sl]), axis=-1)
    return im


def integrate_pred_lab(pred, lab):
    """Integrating prediction and label into two rows of one image"""
    return np.concatenate((pred, lab), axis=-2)


def norm01(x):
    """Normalize all values to range 0 - 1"""
    return (x - x.min()) / (x.max() - x.min())


def make_image(data, outputname, size=(20, 2), dpi=1000):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('jet')

    ax.imshow(data, aspect='equal')
    plt.savefig(outputname, dpi=dpi)
    plt.close()


dir_out = r'F:\BACKUPS\MRA\run005/'
pred = np.load('%sval_pred_best_epoch.npy' % dir_out)
lab = np.load('%sval_lab_best_epoch.npy' % dir_out)

for k in range(pred.shape[0]):
    print(k)
    im = integrate_pred_lab(integrate_slices((pred[k])), integrate_slices(lab[k]))

    make_image(norm01(im), "%spred%02d.png" % (dir_out, k))
    # make_image(im, "%spred%02d.png" % (dir_out, k))

    np.save("%spred%02d.npy" % (dir_out, k), im)





