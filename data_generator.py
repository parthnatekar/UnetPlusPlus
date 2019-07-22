import numpy as np
import keras
from glob import glob
import random
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size = 16, dim = (128, 128), channels = 4, val = False):
        'Initialization'
        self.path = path
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = channels
        self.val = val


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(glob(self.path + '/patches/*.npy')) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Generate data
        #print(indexes_i, indexes_j)

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = []
        y = []

        # Generate data
        while(len(X)) != 16:
            if self.val == True:
                index_i = random.randint(0, 48)
                index_j = random.randint(0, 438)
            else:
                index_i = random.randint(0, 245)
                index_j = random.randint(0, 438)

            if os.path.exists(self.path + 'patches/patch_%d_%d.npy' %(index_i, index_j)) and os.path.exists(self.path + 'masks/label_%d_%d.npy' %(index_i, index_j)):

                # Store sample
                X.append(np.load(self.path + 'patches/patch_%d_%d.npy' %(index_i, index_j)))
                # Store class
                y.append(np.load(self.path + 'masks/label_%d_%d.npy' %(index_i, index_j)))

        #print(np.array(y).shape, np.array(X).shape)

        return np.array(X), np.array(y)

