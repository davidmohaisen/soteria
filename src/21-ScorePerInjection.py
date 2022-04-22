"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import MadryEtAl
from cleverhans.loss import CrossEntropy
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import train, model_eval, tf_model_load, model_argmax
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from six.moves import xrange
from sklearn.model_selection import train_test_split
import keras
from cleverhans.attacks import SPSA
from cleverhans.attacks import SaliencyMapMethod
import pickle
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from statistics import mean
from statistics import stdev
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected
import math
from statistics import mean
from statistics import stdev





tf.reset_default_graph()






# Get MNIST test data



# Train
# x_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_train_Dense.pkl', 'rb')
# y_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/y_train_Dense.pkl', 'rb')
#
# x_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_train_Levels.pkl', 'rb')
# y_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/y_train_Levels.pkl', 'rb')

# Test
# x_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_test_Dense.pkl', 'rb')
# y_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/y_test_Dense.pkl', 'rb')
#
# x_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_test_Levels.pkl', 'rb')
# y_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/y_test_Levels.pkl', 'rb')
# y_true = y_trainFileLevels
# Adv
# x_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XDense.pkl', 'rb')
# y_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialLabelDense.pkl', 'rb')
#
# x_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XLevels.pkl', 'rb')
# y_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialLabelLevels.pkl', 'rb')

# Adv-Classifier
x_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/ClassifierXDense.pkl', 'rb')
y_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/YClassifierLabels.pkl', 'rb')

x_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/ClassifierXLevels.pkl', 'rb')
y_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/YClassifierLabels.pkl', 'rb')
FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/YAttackTrueMajorityLabels.pkl', 'rb')
u = pickle._Unpickler(FileToSave)
u.encoding = 'latin1'
y_true = u.load()

u = pickle._Unpickler(x_trainFileDense)
u.encoding = 'latin1'
x_trainDense = u.load()
u = pickle._Unpickler(y_trainFileDense)
u.encoding = 'latin1'
y_trainDense = u.load()

u = pickle._Unpickler(x_trainFileLevels)
u.encoding = 'latin1'
x_trainLevels = u.load()
u = pickle._Unpickler(y_trainFileLevels)
u.encoding = 'latin1'
y_trainLevels = u.load()

XInjection = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XInjection.pkl', 'rb')
u = pickle._Unpickler(XInjection)
u.encoding = 'latin1'
XInjection = u.load()
arrLabels = list(set(XInjection))



x_trainDense = np.asarray(x_trainDense)
x_trainLevels = np.asarray(x_trainLevels)
x_trainDense = x_trainDense.reshape((len(x_trainDense),500))
x_trainLevels = x_trainLevels.reshape((len(x_trainLevels),500))
x_toTrain = np.concatenate((x_trainDense, x_trainLevels), axis=1)

print(x_toTrain.shape)


num_inputs=1000
num_hid1=2000
num_hid2=3000
num_hid3=2000
num_output=num_inputs

lr=0.0005
actf=tf.nn.relu

X=tf.placeholder(tf.float32,shape=[None,num_inputs])
initializer=tf.variance_scaling_initializer()

w1=tf.Variable(initializer([num_inputs,num_hid1]),dtype=tf.float32)
w2=tf.Variable(initializer([num_hid1,num_hid2]),dtype=tf.float32)
w3=tf.Variable(initializer([num_hid2,num_hid3]),dtype=tf.float32)
w4=tf.Variable(initializer([num_hid3,num_output]),dtype=tf.float32)

b1=tf.Variable(tf.zeros(num_hid1))
b2=tf.Variable(tf.zeros(num_hid2))
b3=tf.Variable(tf.zeros(num_hid3))
b4=tf.Variable(tf.zeros(num_output))

hid_layer1=actf(tf.matmul(X,w1)+b1)
hid_layer2=actf(tf.matmul(hid_layer1,w2)+b2)
hid_layer3=actf(tf.matmul(hid_layer2,w3)+b3)
output_layer=actf(tf.matmul(hid_layer3,w4)+b4)

loss=tf.reduce_mean(tf.square(output_layer-X))

optimizer=tf.train.AdamOptimizer(lr)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

num_epoch=100
batch_size=128

sessAllClasses = tf.Session()
sessPerClass = tf.Session()

sess = tf.Session()
sess = tf.Session()
sessAllClasses.run(init)
sessPerClass.run(init)



#### Classification Stage 2 ####
FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/MeanSTDevDistances.pkl', 'rb')
# FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/MaxDistances.pkl', 'rb')
u = pickle._Unpickler(FileToSave)
u.encoding = 'latin1'
MaxDistances = u.load()

FileToSaveAllClasses = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/AllClassesMeanSTDevDistances.pkl', 'rb')
u = pickle._Unpickler(FileToSaveAllClasses)
u.encoding = 'latin1'
MaxDistancesAllClasses = u.load()


catchAllClasses = []
NotCatchAllClasses = []
All = []
for i in range(len(arrLabels)):
    catchAllClasses.append([0,0,0,0])
    NotCatchAllClasses.append([0,0,0,0])
    All.append([0,0,0,0])

factor = 1.0
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,"/home/ahmed/Documents/Projects/IoT_Defence/AEModels/AllClasses.ckpt")
print("Model AllClasses restored.")
resultsAllClasses=output_layer.eval(session=sess,feed_dict={X:x_toTrain})



for i in range(int(len(x_toTrain)/10)):
    scoreAllClasses = 0
    LabelIndex = (i*10) #In case of any thing not Adv-Classifier
    # LabelIndex = i #In case of Adv-Classifier
    for m in range(10):
        DistanceAllClasses = []
        for j in range(len(x_toTrain[(i*10)+m])):
            valueToAddAllClasses = (x_toTrain[(i*10)+m][j] - resultsAllClasses[(i*10)+m][j])**2
            DistanceAllClasses.append(valueToAddAllClasses)
        L2DistanceAllClasses = (sum(DistanceAllClasses))**(1/2)
        if L2DistanceAllClasses > (MaxDistancesAllClasses[0]+MaxDistancesAllClasses[1]*factor):
            scoreAllClasses += 1

    if scoreAllClasses >= 5 :
        catchAllClasses[arrLabels.index(XInjection[i*10])][y_trainDense[LabelIndex]] += 1
    else:
        NotCatchAllClasses[arrLabels.index(XInjection[i*10])][y_trainDense[LabelIndex]] += 1

    All[arrLabels.index(XInjection[i*10])][y_trainDense[LabelIndex]] += 1

for i in range(len(arrLabels)):
    print(arrLabels[i])
    print(NotCatchAllClasses[i])
    print(catchAllClasses[i])
    print(All[i])
    print(sum(catchAllClasses[i]) / sum(All[i]))
