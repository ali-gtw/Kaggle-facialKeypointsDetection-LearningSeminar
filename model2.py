import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

import cPickle as pickle

from submission import submit


FTRAIN = 'data/training.zip'
FTEST = 'data/test.zip'

def load (test=False, cols=None):
    fname = FTEST if test else FTRAIN
    df = read_csv (fname) # Load pandas dataframe
    
    
    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda  im: np.fromstring(im, sep=' '))
    
    if cols:
        df = df[list(cols) + ['Image']]
        
    #print (df.count())
    
    df = df.dropna() # drop all rows that have missing values in the
    
    X = np.vstack(df['Image'].values) / 255 # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    
    if not test: # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y-48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None
        
    return X, y




from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


def load2d (test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y



net2 = NeuralNet(
    
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    
    
    input_shape = (None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,
    
    
    update_learning_rate=0.01,
    update_momentum=0.9,
    
    regression=True,
    max_epochs=1000,
    verbose=1,
    
    )

X, y = load2d()
net2.fit(X, y)




with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)





submit(net2, load2d)
