
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


os.environ['CUDA_VISIBLE_DEVICES'] = ''



tf.reset_default_graph()


# Adv-Classifier
x_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/ClassifierXDense.pkl', 'rb')
y_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/YClassifierLabels.pkl', 'rb')

x_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/ClassifierXLevels.pkl', 'rb')
y_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/YClassifierLabels.pkl', 'rb')
FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/YAttackTrueMajorityLabels.pkl', 'rb')
u = pickle._Unpickler(FileToSave)
u.encoding = 'latin1'
y_true = u.load()
FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialSizeUsedDense.pkl', 'rb')
u = pickle._Unpickler(FileToSave)
u.encoding = 'latin1'
sizeUsed = u.load()

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
#### Train ####
# for epoch in range(num_epoch):
#     counter = 0
#     num_batches=int(math.ceil(len(x_toTrain)/batch_size))
#     for iteration in range(num_batches):
#         start = counter * batch_size
#         end = start + batch_size
#         if end > len(x_toTrain) :
#             end = len(x_toTrain)
#         X_batch = x_toTrain[start:end]
#         sess.run(train,feed_dict={X:X_batch})
#         counter += 1
#
#     train_loss=loss.eval(feed_dict={X:X_batch})
#     print("epoch {} loss {}".format(epoch,train_loss))
#
# saver = tf.train.Saver()
# save_path = saver.save(sess, "/home/ahmed/Documents/Projects/IoT_Defence/AEModels/"+str(ToSelect)+".ckpt")
# print("Model saved in path: %s" % save_path)




#### Validation : Distance ####
# Distances = [[],[],[],[]]
# for N in range(4):
#     saver = tf.train.Saver()
#     saver.restore(sess, "/home/ahmed/Documents/Projects/IoT_Defence/AEModels/"+str(N)+".ckpt")
#     print("Model "+str(N)+" restored.")
#     results=output_layer.eval(feed_dict={X:x_toTrain})
#     print(results.shape)
#     for i in range(len(x_toTrain)):
#         Distance = []
#         if y_trainDense[i] == N:
#             for j in range(len(x_toTrain[i])):
#                 valueToAdd = (x_toTrain[i][j] - results[i][j])**2
#                 Distance.append(valueToAdd)
#             L2Distance = (sum(Distance))**(1/2)
#             Distances[N].append(L2Distance)
# maxDistances = [0,0,0,0]
# for i in range(4):
#     maxDistances[i] = [mean(Distances[i]),stdev(Distances[i])]
#
#
# print(maxDistances)
# FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/MeanSTDevDistances.pkl', 'wb')
# # FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/MaxDistances.pkl', 'wb')
# pickle.dump(maxDistances, FileToSave)
# exit()




#### Validation : Distance All ####
# Distances = []
# saver = tf.train.Saver()
# saver.restore(sess, "/home/ahmed/Documents/Projects/IoT_Defence/AEModels/AllClasses.ckpt")
# print("Model restored.")
# results=output_layer.eval(feed_dict={X:x_toTrain})
# print(results.shape)
# for i in range(len(x_toTrain)):
#     Distance = []
#     for j in range(len(x_toTrain[i])):
#         valueToAdd = (x_toTrain[i][j] - results[i][j])**2
#         Distance.append(valueToAdd)
#     L2Distance = (sum(Distance))**(1/2)
#     Distances.append(L2Distance)
#
# # maxDistances = [mean(Distances),stdev(Distances)]
# maxDistances = max(Distances)
#
#
# print(maxDistances)
# # FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/AllClassesMeanSTDevDistances.pkl', 'wb')
# FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/AllClassesMaxDistances.pkl', 'wb')
# pickle.dump(maxDistances, FileToSave)
# exit()




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

catch = [0,0,0,0]
catchOnlyN = [0,0,0,0]
catchAllClasses = [0,0,0,0]
All = [0,0,0,0]
mal2ben = 0
factor = 1.0
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,"/home/ahmed/Documents/Projects/IoT_Defence/AEModels/AllClasses.ckpt")
print("Model AllClasses restored.")
resultsAllClasses=output_layer.eval(session=sess,feed_dict={X:x_toTrain})
toTest = []
Labels = []
Size = []
for N in range(4):
    for i in range(int(len(x_toTrain)/10)):
        score = 0
        scoreAllClasses = 0
        LabelIndex = (i*10) #In case of any thing not Adv-Classifier
        # LabelIndex = i #In case of Adv-Classifier
        if y_trainDense[LabelIndex] == N :
            for m in range(10):
                Distance = []
                DistanceAllClasses = []
                for j in range(len(x_toTrain[(i*10)+m])):
                    # valueToAdd = (x_toTrain[(i*10)+m][j] - results[(i*10)+m][j])**2
                    valueToAddAllClasses = (x_toTrain[(i*10)+m][j] - resultsAllClasses[(i*10)+m][j])**2
                    # Distance.append(valueToAdd)
                    DistanceAllClasses.append(valueToAddAllClasses)
                # L2Distance = (sum(Distance))**(1/2)
                L2DistanceAllClasses = (sum(DistanceAllClasses))**(1/2)
                if L2DistanceAllClasses > (MaxDistancesAllClasses[0]+MaxDistancesAllClasses[1]*factor):
                    scoreAllClasses += 1
                # if L2Distance > (MaxDistances[y_trainDense[LabelIndex]][0]+MaxDistances[y_trainDense[LabelIndex]][1]*factor):
                #     score += 1
            if score >= 5  or scoreAllClasses >= 5:
                catch[y_trainDense[LabelIndex]] += 1
                # if score >= 5 :
                #
            else :
                toTest.append(x_toTrain[(i*10):(i*10)+10])
                Labels.append(y_true[i])
                Size.append(sizeUsed[i])
print(len(toTest))
print(len(toTest[0]))
print(len(Labels))
# FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/SamplesPassesDetector/X.pkl', 'wb')
# pickle.dump(toTest, FileToSave)
# FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/SamplesPassesDetector/Y.pkl', 'wb')
# pickle.dump(Labels, FileToSave)
FileToSave = open('/home/ahmed/Documents/Projects/IoT_Defence/SamplesPassesDetector/Size.pkl', 'wb')
pickle.dump(Size, FileToSave)
