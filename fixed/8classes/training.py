#!/usr/bin/env python3
## import libraries
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, TimeDistributed, RepeatVector
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

# set GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# specify
name = sys.argv[1]

n_step = 9

def split(data, depth):
    dataset = data
    depth = depth
    X = np.zeros([int(depth), n_step, 3])
    Y = np.zeros([int(depth), n_step, 3])
    c = 0
    d = 0
    
    for i in range(int(depth)):
        for j in range(n_step):
            if d < dataset.shape[0]:
                X[i,j,:] = dataset[d, c:c+3]
                Y[i,j,:] = dataset[d, (c+3*n_step):(c+3*n_step+3)]
                
                if ((c+3*n_step+3) != (dataset.shape[1])):
                    c +=3
                else:
                    c = 0
                    d += 1
        if (c-3) > 0:
            c = (c - 3*n_step + 3)
        else:
            c = c
            
    return X, Y

path = '/home/lab606a/Documents/20200331/fixed/original data 60hz/' + name + '.csv'
print('read dataset ' + path)
dataset = pd.read_csv(path, header=None)
dataset = dataset.fillna(0)
dataset = np.array(dataset)
dataset.shape

maxlen_train = dataset.shape[1]+((n_step-1)*3)

dataset = sequence.pad_sequences(dataset, maxlen=maxlen_train, padding='post', dtype='float32')
dataset.shape

depth_train = (int(dataset.shape[1]/3)+1-n_step-n_step)*dataset.shape[0] # (all_balls + 1 - input_balls - output_balls)*n_rows

x_train, y_train = split(data=dataset, depth=depth_train)

model = Sequential()
model.add(LSTM(512, activation='linear', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(RepeatVector(x_train.shape[1]))
model.add(LSTM(512, activation='linear', return_sequences=True))
model.add(TimeDistributed(Dense(3)))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=3000, epochs=500, shuffle=True)

# plot accuracy history
# summarize history for accuracy 
plt.plot(history.history['acc'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
acc_png = './history/' + name + '_accuracy.png'
plt.savefig(acc_png)

# saved model
model_path = './saved model/60hz_9step_512/prediction_' + name 
model.save(model_path)
