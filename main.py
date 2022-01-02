import os
from keras import layers, optimizers, models, losses, metrics, regularizers, applications
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import random
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from PIL import Image
from keras.utils import plot_model
import tensorflow as tf
import keras
from keras.utils import to_categorical
import threading
from datagen import data_gen


px = 256
timesteps = 3

def time_dist(inputs):
    x = layers.TimeDistributed(applications.InceptionV3(include_top=False, weights='imagenet'))(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    return x


inputs = layers.Input(shape=(3, 256, 256, 3))
time_layer = layers.Input(shape=(3,))
mod = time_dist(inputs)


permute = layers.Permute((2, 1))(mod)
interval_scaling = layers.Multiply()([permute, time_layer])
mod = layers.Permute((2, 1))(interval_scaling)
mod = layers.Activation('relu')(mod)
mod=layers.BatchNormalization(axis=-1)(mod)
mod = layers.Dropout(0.4)(mod)
out = layers.GRU(1, activation='sigmoid', 
                            kernel_regularizer=regularizers.L1L2(l1=0, l2=2e-5))(mod)
model = models.Model([inputs, time_layer], out)
print(model.summary())

model.compile(
    loss=losses.binary_crossentropy,
    optimizer=optimizers.Adam(1e-4),
    metrics=[metrics.AUC(name='auc'),
             metrics.SensitivityAtSpecificity(0.9, name='sens'),
             metrics.SpecificityAtSensitivity(0.9, name='spec'),]
)



batch_size = 32
train_generator = data_gen(
    "../BMJOO/Data/train", batch_size, px=256)

val_generator = data_gen(
    "../BMJOO/Data/val", batch_size, px=256, preprocess=None)

train_steps = 2942// batch_size
val_steps = 981 // batch_size


reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_auc", factor=0.67, patience=3, verbose=1, mode="max", epsilon=0.0001)


earlyStopping = EarlyStopping(
    monitor='val_auc', patience=15, verbose=0, mode='max')
mcp_save = ModelCheckpoint(
    'model.h5', save_best_only=True, monitor='val_auc', mode='max', verbose=1)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=250,
    verbose=1,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[reduceLROnPlat,
               earlyStopping,
               mcp_save],
    class_weight={0: 1, 1: 9.81617647},
)


###############################################################################
import os
import numpy as np
from keras import layers, models
from custom_metrics import brier, sens, spec, HL, auc_roc, bal, net 
from fast_delong import delong_roc_test
from datagen_both import data_gen
import sklearn

def extract_true(files):
    i=0
    true = np.zeros((980))
    n = sorted(os.listdir(files))
    while i<980:
        if 'no' not in n[i]: 
            true[i] = 1
        i+=1
    return true

true = extract_true("../BMJOO/Data/tvt_new/test")

batch_size = 1
test_steps = 980//batch_size
test_generator = data_gen(
    "../BMJOO/Data/tvt_new/test", 1, px=256, preprocess=None)


inputs = layers.Input(shape=(3, 256, 256, 3))
time_layer = layers.Input(shape=(3,))
mod = time_dist(inputs)
permute = layers.Permute((2, 1))(mod)
interval_scaling = layers.Multiply()([permute, time_layer])
mod = layers.Permute((2, 1))(interval_scaling)
mod = layers.Activation('relu')(mod)
mod=layers.BatchNormalization(axis=-1)(mod)
mod = layers.Dropout(0.4)(mod)
out = layers.GRU(1, activation='sigmoid', 
                            kernel_regularizer=regularizers.L1L2(l1=0, l2=2e-5))(mod)
model = models.Model([inputs, time_layer], out)
print(model.summary())

model.compile(
    loss=losses.binary_crossentropy,
    optimizer=optimizers.Adam(1e-4),
    metrics=[metrics.AUC(name='auc'),
             metrics.SensitivityAtSpecificity(0.9, name='sens'),
             metrics.SpecificityAtSensitivity(0.9, name='spec'),]
)



model.load_weights('model.h5')



evaluate = model.evaluate(
                    test_generator,
                    steps=172,
                    verbose=1
                    )

predict = model.predict_generator(
                    test_generator,
                    steps=172,
                    verbose=1
                    )
