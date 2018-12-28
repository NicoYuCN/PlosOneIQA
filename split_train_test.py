import numpy as np
#import hickle as hkl
#import scipy.misc
#from sklearn.cross_validation import StratifiedShuffleSplit,ShuffleSplit,train_test_split
#from pylearn2.utils.rng import make_np_rng
import os
import sys
import h5py
#import shutil
def split_train_test(train_index,test_index,label):
   train_label=[]
   train_image=[]
   test_label=[]
   test_image=[]
   image_all_lcn=np.load('./data/image_all/image_all_lcn.npy')#note lcn is float
   #image_all_lcn = hkl.load('./data/image_all/image_all.hkl')#this is no normalization
   print('image_all.shape',image_all_lcn.shape)
   print('train_index',train_index)
   print('test_index' ,test_index)
   #train_index = train_index[np.argsort(train_index)]
   #test_index = test_index[np.argsort(test_index)]
   #print('train_index_sort',train_index)
   #print('test_index_sort' ,test_index)
   for i in train_index:
       train_image.append(image_all_lcn[:,:,:,i])
       train_label.append(label[i])
   train_image = np.asarray(train_image)
   train_image = np.rollaxis(train_image,0,4)#(number,3,h,w)->(3,h,w,number)
   print('train_image.shape: ',train_image.shape)
   np.save('./data/labels/train_label.npy',train_label)
   np.save('./data/train/train_image.npy',train_image)
   for i in test_index:
       test_image.append(image_all_lcn[:,:,:,i])
       test_label.append(label[i])
   test_image = np.asarray(test_image)
   test_image = np.rollaxis(test_image,0,4)#(number,3,h,w)->(3,h,w,number)
   print('test_image.shape: ',test_image.shape)
   np.save("./data/labels/test_label.npy",test_label)
   np.save('./data/test/test_image.npy',test_image)
   print('Split Done.....')

if __name__ == '__main__':
    EXP_ID = sys.argv[1] if len(sys.argv) > 1 else None
    #Info = h5py.File('TID2008info.mat')
    #Info = h5py.File('LIVEinfo.mat')
    Info = h5py.File('BIDinfo.mat')
    index = Info['index'][:, int(EXP_ID)%1000]
    ref_ids = Info['ref_ids'][:,0]
    img_ids = Info['img_ids'][:,0]
    trainindex = index[0:int(np.ceil(4.0*len(index)/5))]
    #testindex = index[4*len(index)//5:len(index)]
    train_index = []
    test_index = []
    for i in range(len(ref_ids)):
        if ref_ids[i] in trainindex:
            train_index.append(int(img_ids[i])-1)
        else:
            test_index.append(int(img_ids[i])-1)
    if os.path.isdir('./data/train')==False:
       os.makedirs('./data/train')
    if os.path.isdir('./data/test') ==False:
       os.makedirs('./data/test')
    label=np.load('./data/labels/label_all.npy')
    print('label_all.shape',label.shape)
    split_train_test(train_index, test_index, label)
