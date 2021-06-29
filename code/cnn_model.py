import datetime
import shutil

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pickle
import os
import tensorflow as tf

WEBSITES = 0
DATASET_PATH = r"D:\wfp\QUIC-TOR"
EXPERIMENT_FILE = "0_100_split"

def bernoulli(split_rate):
    if tf.random.uniform(shape=()).numpy() > split_rate:
        return False
    else:
        return True


def quic_cnn():
    global WEBSITES, DATASET_PATH
    es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
    train_db, test_db = load_big_data(DATASET_PATH)
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(log_dir)
    shutil.copy(__file__, log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    kernel_size = 5
    pool_size = 4
    dropout = 0.1
    lr = 0.001
    max_features = 1
    maxlen = 5000

    model = Sequential()
    model.add(Reshape((5000, 1), input_shape=(1, 5000, 1)))
    model.add(Dropout(input_shape=(maxlen, max_features), rate=dropout))

    model.add(Conv1D(filters=128,
                     kernel_size=kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=pool_size, padding='valid'))

    model.add(Conv1D(filters=64,
                     kernel_size=kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=pool_size, padding='valid'))

    model.add(Conv1D(filters=32,
                     kernel_size=kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=pool_size, padding='valid'))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(WEBSITES, activation='softmax'))

    optimizer = RMSprop(lr=lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(train_db, epochs=50, callbacks=[tensorboard_callback, es])
    result = model.evaluate(test_db)
    with open(os.path.join(log_dir, "acc.txt"), "w") as f:
        f.write(f"loss and acc: {result}")


def load_big_data(path, split_rate=0.8):
    global WEBSITES, EXPERIMENT_FILE
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
            if bernoulli(split_rate):
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
    train_db = train_db.shuffle(buffer_size=8192)
    train_db = train_db.batch(128, drop_remainder=True)
    train_db = train_db.prefetch(tf.data.experimental.AUTOTUNE)

    test_db = test_db.map(read_trace)
    test_db = test_db.batch(128, drop_remainder=True)

    return train_db, test_db


def read_trace(x, y):
    def _inner(x):
        data = np.load(x.numpy().decode())["arr_0"]
        data = data[:5000]
        data = data.reshape([5000, 1])
        return data

    data = tf.py_function(_inner, [x], [tf.float32])
    return data, y


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    quic_cnn()
