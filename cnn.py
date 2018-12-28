# -*- coding: utf-8 -*-
import os,sys
os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32"
import time
#import cPickle
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances
#import hickle as hkl
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
#from keras.utils.visualize_util import plot
#from keras.layers.core import Layer
#from keras.regularizers import l2
#from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from keras.callbacks import ModelCheckpoint

def load_data():
  """
  Load dataset and split data into training and test sets
  """
  #-------train part------
  X_train=np.load('data/train/train_image.npy')
  #-------test part-------
  X_test=np.load('data/test/test_image.npy')
  #---------train_labels-----
  Y_train=np.load('data/labels/train_label_patch.npy')
  #---------test_labels-----
  Y_test=np.load('data/labels/test_label_patch.npy')
  print('X_train.shape: ',X_train.shape)
  print('X_test.shape: ',X_test.shape)
  return (X_train,X_test,Y_train,Y_test)

def CNN_model(img_rows, img_cols, channel=3, num_class=None):

    input = Input(shape=(channel, img_rows, img_cols))
    conv1_7x7= Convolution2D(16,7,7,name='conv1/7x7',activation='relu')(input)#,W_regularizer=l2(0.0002)
    pool2_2x2= MaxPooling2D(pool_size=(2,2),strides=(1,1),border_mode='valid',name='pool2')(conv1_7x7)
    poll_flat = Flatten()(pool2_2x2)
    #MLP
    fc_1 = Dense(200,name='fc_1',activation='relu')(poll_flat)
    drop_fc = Dropout(0.5)(fc_1)
    out = Dense(1,name='fc_2',activation='sigmoid')(drop_fc)
    # Create model
    model = Model(input=input, output=out)
    # Load cnn pre-trained data 
    #model.load_weights('models/weights.h5')#NOTE 
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mean_absolute_error')  
    return model
"""
def grnn(F_test,sigma, F_train, Y_train):
    Dist = np.exp(-euclidean_distances(F_test,F_train)/(2*sigma**2))
    predictions_grnn = []
    for i in range(F_test.shape[0]):
        predictions_grnn.append(np.sum(np.dot(Dist[i,:], Y_train))/np.sum(Dist[i,:]))
    return predictions_grnn
"""

if __name__ =='__main__':
    EXP_ID = sys.argv[1] if len(sys.argv) > 1 else None #
    batch_size=100
    nb_epoch =200
    # TODO: Load training and test sets
    X_train, X_test, Y_train, Y_test = load_data()
    model = CNN_model(16, 16, 3)
    #plot(model, to_file='model.png')
    weight_path = 'models/%s' % (EXP_ID)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    checkpointer=ModelCheckpoint(filepath='%s/weights.h5'%weight_path,monitor='val_loss',verbose=1,save_best_only=True)

    #start_time = time.clock()
    model.fit(X_train,Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=2, #
              callbacks=[checkpointer],
              validation_split=0.2
              )
    end_time = time.clock()

    #print('train time is: ',(end_time-start_time))
    print('load best weight.......')
    model.load_weights('%s/weights.h5'%weight_path)
    #cPickle.dump(model,open("./models/model.pkl","wb"))#save best model+weight
    #if other code want use this final model run: model = cPickle.load(open("model.pkl","rb"))
    #start_time = time.clock()
    print("make predictions.......")
    predictions_test = model.predict(X_test, batch_size=batch_size, verbose=1)
    #end_time = time.clock()
    #print('test time is: ',(end_time-start_time))
    print("\nmake average.....")
    predictions_test=np.asarray(predictions_test)
    result=[]
    b = np.load('./data/labels/test_label.npy')
    k = predictions_test.shape[0]/b.shape[0]

    for i in range(b.shape[0]):
        result.append(np.average(predictions_test[i*k:(i+1)*k,0]))
    np.save('./data/labels/test_result.npy',result)#predict result
    PLCC = stats.pearsonr(result, b)[0]
    SROCC = stats.spearmanr(result, b)[0]
    RMSE = sqrt(mean_squared_error(result, b))
    #OR = (abs(np.array(result)-np.array(b))>2*np.array(mos_std_test)).mean()
    KROCC= stats.stats.kendalltau(result, b)[0]
    print("MLP: SROCC: %s PLCC: %s RMSE: %s KROCC: %s"%(SROCC,PLCC,RMSE,KROCC))

    """
    trimmed_model = Model(input=model.input, output=model.get_layer('fc_1').output)
    features_train = trimmed_model.predict(X_train, batch_size=batch_size, verbose=2)
    features = trimmed_model.predict(X_test, batch_size=batch_size, verbose=2)
    features_train = np.asarray(features_train)
    features = np.asarray(features)
    M = features_train.max(axis=1)
    for i in range(features_train.shape[0]):
        if M[i]>0:
            features_train[i,:] = features_train[i,:]/M[i]
    M = features.max(axis=1)
    for i in range(features.shape[0]):
        if M[i]>0:
            features[i,:] = features[i,:]/M[i]
    sigma = 0.01
    start_time = time.clock()
    predictions_grnn = grnn(features, sigma, features_train, Y_train)
    end_time = time.clock()
    print('\ntest time is: ', (end_time - start_time))
    predictions_grnn = np.asarray(predictions_grnn)
    result = []
    for i in range(b.shape[0]):
        result.append(np.average(predictions_grnn[i*k:(i+1)*k]))
    np.save('./data/labels/test_result.npy', result)  # predict result
    PLCC = stats.pearsonr(result, b)[0]
    SROCC = stats.spearmanr(result, b)[0]
    print("GRNN: SROCC: %s PLCC: %s" % (SROCC, PLCC))
    """

    print("Done....")
