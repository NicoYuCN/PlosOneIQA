#import hickle as hkl
import numpy as np
#import scipy.misc


def patch_image(image,label,size):
    patch_data = []
    patch_label = []
    for k in range(image.shape[3]):
       for j in range(image.shape[2]/size):
           for i in range(image.shape[1]/size):
                patch_data.append(image[:,i*size:(i+1)*size,j*size:(j+1)*size,k])
                patch_label.append(label[k])
                #print('i,j,k: ',i,j,k)
    x=np.asarray(patch_data)
    y=np.asarray(patch_label)
    return x,y


if __name__ == '__main__':
    train_label=np.load('./data/labels/train_label.npy')
    test_label=np.load('./data/labels/test_label.npy')
    train_image= np.load('./data/train/train_image.npy')
    test_image= np.load('./data/test/test_image.npy')
    print('train_image.shape: ',train_image.shape)
    print('test_image.shape: ',test_image.shape)

    train_patch,train_label =patch_image(train_image,train_label,16)
    print('train_image.shape: ',train_patch.shape)
    np.save('./data/train/train_image.npy',train_patch)
    np.save('./data/labels/train_label_patch.npy',train_label)
    test_patch,test_label =patch_image(test_image,test_label,16)
    print('test_image.shape: ',test_patch.shape)
    np.save('./data/test/test_image.npy',test_patch)
    np.save('./data/labels/test_label_patch.npy',test_label)
    print('Patch Done.....')