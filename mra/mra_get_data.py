"""Brain MRA"""
import glob
from scipy.io import loadmat
import pylab as plt
from time import time
import numpy as np
import pickle

read_npy = True
save_list = True
dir_in = r"C:\Workspace\KU AIS Patient Anonymized/"
files = ['ColArt', 'ColCap', 'ColDel', 'ColEVen', 'ColLVen', 'IMG']

folders = glob.glob("%s/IAT AIS_pros/*/*DSC*/" % dir_in) + glob.glob("%s/IAT AIS_pros/*/*PRE*/*DSC*/" % dir_in)

print(folders.__len__())
print(folders)

start = time()
subject_list = []
for (i, folder) in enumerate(folders):
    print(i, folder)
    for file in files:
        if read_npy:
            x = np.load('%s%s.npy' % (folder, file))
        else:
            x = loadmat('%s%s.mat' % (folder, file))
            x = x[file]
            np.save('%s%s.npy' % (folder, file), x.transpose((2, 3, 0, 1)))
        print(x.shape)
        if np.prod(x.shape) == (240 * 240 * 60 * 20):
            subject_list.append(folder)

if save_list:
    with open("%ssubject_list.txt" % dir_in, "wb") as fp:   #Pickling
        pickle.dump(subject_list, fp)
print(time() - start)

