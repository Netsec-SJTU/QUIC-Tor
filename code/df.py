import datetime
import os
import pickle
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Reshape
from tensorflow.keras.layers import ELU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax

WEBSITES = -1

DATASET_DIR = r"D:\wfp\crawle"
EXPERIMENT_FILE = "0_open_mix_200_200_30000"
EXPERIMENT_FILE_PATH = os.path.join(DATASET_DIR, EXPERIMENT_FILE)


def bernoulli():
    return tf.random.uniform(shape=()).numpy()


def load_big_data(path, split_rate=0.8):
    global WEBSITES
    list_file = os.path.join(path, EXPERIMENT_FILE)
    with open(list_file, "rb") as f:
        sites_list = pickle.load(f)
    WEBSITES = len(sites_list)

    x_train = list()
    y_train = list()
    x_test = list()
    y_test = list()
    for index in range(len(sites_list)):
        for file_name in sites_list[index]:
            rand_value = bernoulli()
            if 0 <= rand_value <= split_rate:
                x_train.append(os.path.join(path, file_name))
                y_train.append(index)
            else:
                x_test.append(os.path.join(path, file_name))
                y_test.append(index)
    x_train = tf.constant(x_train)
    y_train = tf.constant(y_train)
    x_test = tf.constant(x_test)
    y_test = tf.constant(y_test)

    y_train = tf.one_hot(y_train, depth=WEBSITES)
    y_test = tf.one_hot(y_test, depth=WEBSITES)

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_db = train_db.map(
        map_func=read_trace,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_db = train_db.shuffle(buffer_size=2048)
    train_db = train_db.batch(32, drop_remainder=True)
    train_db = train_db.prefetch(tf.data.experimental.AUTOTUNE)

    test_db = test_db.map(read_trace)
    test_db = test_db.batch(32, drop_remainder=True)


    return train_db, test_db


def read_trace(x, y):
    def _inner(x):
        data = np.load(x.numpy().decode())["arr_0"]
        data = data[:5000]
        # data = data.reshape([5000, 1])
        return data

    data = tf.py_function(_inner, [x], [tf.float32])
    return data, y


class DFNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #Block1
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]
        model.add(Reshape((5000,1),input_shape=(1,5000)))
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act1'))
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding='same', name='block1_pool'))
        model.add(Dropout(0.1, name='block1_dropout'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act1'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                               padding='same', name='block2_pool'))
        model.add(Dropout(0.1, name='block2_dropout'))

        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act1'))
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                               padding='same', name='block3_pool'))
        model.add(Dropout(0.1, name='block3_dropout'))

        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act1'))
        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                               padding='same', name='block4_pool'))
        model.add(Dropout(0.1, name='block4_dropout'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(1024, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc1_act'))


        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc2_act'))


        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc3'))
        model.add(Activation('softmax', name="softmax"))
        return model


def main():
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(log_dir)
    shutil.copy(__file__, log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6)
    train_db, test_db = load_big_data(DATASET_DIR)
    shutil.copy(EXPERIMENT_FILE_PATH, log_dir)
    model = DFNet.build(5000, WEBSITES)
    optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(train_db, epochs=50, callbacks=[tensorboard_callback, es])
    result = model.evaluate(test_db)
    with open(os.path.join(log_dir, "acc.txt"), "w") as f:
        f.write(f"test loss and test acc: {result}\n")


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    main()
