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



# Test
x_trainFileDense2 = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_test_Dense.pkl', 'rb')
x_trainFileLevels2 = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_test_Levels.pkl', 'rb')



u = pickle._Unpickler(x_trainFileDense2)
u.encoding = 'latin1'
x_trainDense2 = u.load()
u = pickle._Unpickler(x_trainFileLevels2)
u.encoding = 'latin1'
x_trainLevels2 = u.load()




x_trainDense2 = np.asarray(x_trainDense2)
x_trainLevels2 = np.asarray(x_trainLevels2)


x_trainDense2 = x_trainDense2.reshape((len(x_trainDense2),500))
x_trainLevels2 = x_trainLevels2.reshape((len(x_trainLevels2),500))


X_Clean = np.concatenate((x_trainDense2, x_trainLevels2), axis=1)

print(X_Clean.shape)


##### Dont Touch Below #####
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
FileToSaveAllClasses = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/AllClassesMeanSTDevDistances.pkl', 'rb')
u = pickle._Unpickler(FileToSaveAllClasses)
u.encoding = 'latin1'
MaxDistancesAllClasses = u.load()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,"/home/ahmed/Documents/Projects/IoT_Defence/AEModels/AllClasses.ckpt")
print("Model AllClasses restored.")
##### Dont Touch Above #####

CleanResults=output_layer.eval(session=sess,feed_dict={X:X_Clean})

print(CleanResults.shape)

CleanDistances = []
CleanDistancesAVG = []
for i in range(int(len(X_Clean)/10)):
    LabelIndex = (i*10) #In case of any thing not Adv-Classifier
    allDistances = []
    for m in range(10):
        DistanceAllClasses = []
        for j in range(len(X_Clean[(i*10)+m])):
            valueToAddAllClasses = (X_Clean[(i*10)+m][j] - CleanResults[(i*10)+m][j])**2
            DistanceAllClasses.append(valueToAddAllClasses)
        L2DistanceAllClasses = (sum(DistanceAllClasses))**(1/2)
        CleanDistances.append(L2DistanceAllClasses)
        allDistances.append(L2DistanceAllClasses)
    CleanDistancesAVG.append(mean(allDistances))

print("Clean : ",CleanDistances)
print("Clean Avg: ",CleanDistancesAVG)

FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/Graphs/DistanceParameters/CleanTest.pkl', 'wb')
pickle.dump(CleanDistances, FileToSave)
FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/Graphs/DistanceParameters/CleanAvgTest.pkl', 'wb')
pickle.dump(CleanDistancesAVG, FileToSave)
