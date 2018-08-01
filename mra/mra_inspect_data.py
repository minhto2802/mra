"""Brain MRA"""
import glob
from scipy.io import loadmat
import pylab as plt

dir_in = r"C:\Workspace\IAT AIS_pros\BP15 2017081" \
         r"0 R M1 occ\AX_PWI_DSC_Collateral/"
files = ['ColArt', 'ColCap', 'ColDel', 'ColEVen', 'ColLVen', 'IMG']
fig_idx = 0
to_plt = 0
for file in files[-1:]:
    a = loadmat('%s%s.mat' % (dir_in, file))
    b = a[file]
    print(b.shape, b.min(), b.max())
    if to_plt:
        plt.figure()
        for i in range(20):
            plt.subplot(4, 5, i+1)
            plt.imshow(b[..., 2, i], cmap='gray',
                       vmin=b.min(), vmax=b.max())
        fig_idx += 1
    plt.plot(b[88, 138, :, 12])
plt.show()
print