import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import matplotlib.pyplot as plt

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




net1 = NeuralNet(
    # defining layers, three layers 
    layers=[
        ('input' , layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    
    
    input_shape = (None, 9216),  # 96*96 input pixels per batch
    hidden_num_units = 100, # number of units in hidden layer
    output_nonlinearity = None, # output  layer uses identity function
    
    
    output_num_units = 30,  # 30 target values
    
    
    # optimization method
    update = nesterov_momentum,
    update_learning_rate = 0.01,
    update_momentum = 0.9,
    
    
    regression = True, # a flag indicating we're dealing with regression problem
    max_epochs = 1, # we want to train this many epochs
    verbose=1,
    
    
    
    )




X, y = load()
net1.fit(X, y)





def learning_curve():

    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])

    plt.plot(train_loss, linewidth=3, label="train")
    plt.plot(valid_loss, linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(1e-3, 1e-2)
    plt.yscale("log")
    plt.show()



def plot_sample_test_data (net) :

    def plot_sample (x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

    X, _   = load(test=True)
    y_pred = net.predict(X)


    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)
        
    plt.show()







    #np.savetxt("foo.csv", y_pred, delimiter=",")

submit(net1, load)