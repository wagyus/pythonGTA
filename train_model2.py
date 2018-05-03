# train_model.py

import numpy as np
from alexnet import alexnet
import tensorflow as tf

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'motorcycle-{}-{}-{}-20-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

hm_data = 22
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        train_data = np.load('D:/SelfDrivingGTA/pygta5-motorcycle-training-data-and-model/training_data/training_data-{}-balanced.npy'.format(i))

        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)



# tensorboard --logdir=foo:log --host=localhost --port=6006
