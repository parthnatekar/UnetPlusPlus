import numpy as np
import random
import json
from glob import glob
from keras.models import model_from_json, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
import keras.backend as K
from losses import *
# from keras.utils.visualize_util import plot
from extract_patches import *
from data_generator import DataGenerator
from dense_unet import Dense_Unet

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))


class SGDLearningRateTracker(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % 20 == 0 and epoch != 0:
            optimizer = self.model.optimizer
            lr = K.get_value(optimizer.lr)
            decay = K.get_value(optimizer.decay)
            lr = lr / 10
            decay = decay * 10
            K.set_value(optimizer.lr, lr)
            K.set_value(optimizer.decay, decay)
            print('LR changed to:', lr)
            print('Decay changed to:', decay)


class Training(object):

    def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        # loading model from path to resume previous training without recompiling the whole model
        if load_model_resume_training is not None:
            self.model = load_model(load_model_resume_training, custom_objects={'gen_dice_loss': gen_dice_loss,
                                                                                'dice_whole_metric': dice_whole_metric,
                                                                                'dice_core_metric': dice_core_metric,
                                                                                'dice_en_metric': dice_en_metric})
            print("pre-trained model loaded!")
        else:
            unet = Dense_Unet(img_shape=(128, 128, 4))
            self.model = unet.compile_dense()
            #self.model.load_weights('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/U_densenet/U_densenet.25_0.573.hdf5')
            print("U-net CNN compiled!")

    def fit_unet(self, train_gen, val_gen):

        train_generator = train_gen
        val_generator = val_gen
        checkpointer = ModelCheckpoint(
            filepath='/checkpoints/Xnet/X_net{epoch:02d}_{val_loss:.3f}.hdf5',
            verbose=1, period=5)
        self.model.fit_generator(train_generator,
                                 epochs=self.nb_epoch, steps_per_epoch=100, validation_data=val_generator,
                                 validation_steps=100, verbose=1,
                                 callbacks=[checkpointer, SGDLearningRateTracker()])

        # del checkpointer
        # K.clear_session()
        # self.model.fit(X33_train,Y_train, epochs=self.nb_epoch,batch_size=self.batch_size,validation_data=(X_patches_valid,Y_labels_valid),verbose=1, callbacks = [checkpointer,SGDLearningRateTracker()])
        # self.model.fit(X33_train,Y_train, epochs=self.nb_epoch,batch_size=self.batch_size,validation_data=(X_patches_valid,Y_labels_valid),verbose=1, callbacks = [checkpointer,SGDLearningRateTracker()])

    def img_msk_gen(self, X33_train, Y_train, seed):

        '''
        a custom generator that performs data augmentation on both patches and their corresponding targets (masks)
        '''
        datagen = ImageDataGenerator(horizontal_flip=True, data_format="channels_last")
        datagen_msk = ImageDataGenerator(horizontal_flip=True, data_format="channels_last")
        image_generator = datagen.flow(X33_train, batch_size=self.batch_size, seed=seed)
        y_generator = datagen_msk.flow(Y_train, batch_size=self.batch_size, seed=seed)
        while True:
            yield (image_generator.next(), y_generator.next())

    def save_model(self, model_name):
        '''
        INPUT string 'model_name': path where to save model and weights, without extension
        Saves current model as json and weights as h5df file
        '''

        model_tosave = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        json_string = self.model.to_json()
        self.model.save_weights(weights)
        with open(model_tosave, 'w') as f:
            json.dump(json_string, f)
        print('Model saved.')

    def load_model(self, model_name):
        '''
        Load a model
        INPUT  (1) string 'model_name': filepath to model and weights, not including extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        '''
        print('Loading model {}'.format(model_name))
        model_toload = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        with open(model_toload) as f:
            m = next(f)
        model_comp = model_from_json(json.loads(m))
        model_comp.load_weights(weights)
        print('Model loaded.')
        self.model = model_comp
        return model_comp


if __name__ == "__main__":
    # set arguments

    # reload already trained model to resume training
    model_to_load = "Models/ResUnet.04_0.646.hdf5"
    # save=None

    # compile the model
    brain_seg = Training(batch_size=16, nb_epoch=100)

    print("number of trainabale parameters:", brain_seg.model.count_params())
    # print(brain_seg.model.summary())
    # plot(brain_seg.model, to_file='model_architecture.png', show_shapes=True)

    # brain_seg.model.save('models/unet_with_res/unet_with_res.h5')
    print(brain_seg.model.summary())

    train_generator = DataGenerator('/home/brats/parth/parth/_train/', batch_size=32)
    val_generator = DataGenerator('/home/brats/parth/parth/_val/', batch_size=32, val=True)

    #brain_seg.model.save('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/U_densenet/U_densenet.h5')

    brain_seg.fit_unet(train_generator, val_generator)
    #random.seed(7)
'''
    for i in range(7, 25):
        try:
            print('Epoch: ', i)
            train_index = random.randint(0, 25)
            val_index = random.randint(0,5)
            #load data from disk
            Y=np.load("/media/parth/DATA/brats_as_npy_low_res/train_2/y_dataset_{}.npy".format(train_index)).astype(np.uint8)
            X=np.load("/media/parth/DATA/brats_as_npy_low_res/train_2/x_dataset_{}.npy".format(train_index)).astype(np.float32)
            Y_labels_valid=np.load("/media/parth/DATA/brats_as_npy_low_res/val_2/y_dataset_{}.npy".format(val_index)).astype(np.uint8)
            X_patches_valid=np.load("/media/parth/DATA/brats_as_npy_low_res/val_2/x_dataset_{}.npy".format(val_index)).astype(np.float32)
            #print("loading patches done\n")

            # fit model
            brain_seg.fit_unet(X,Y,X_patches_valid,Y_labels_valid, iteration=i)
            del X, Y, X_patches_valid, Y_labels_valid

            gc.collect()
            #t = timeit.timeit('build()', number=1, setup="from __main__ import build")
            mem = get_mem_usage()
            print('mem: {}'.format(mem))
            #if save is not None:
            #    brain_seg.save_model('models/' + save)
        except Exception as e:
            print(e)
            pass
            '''




