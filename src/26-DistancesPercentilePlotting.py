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
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import spline





tf.reset_default_graph()

# Get MNIST test data
x_trainFileDense1 = open('/home/ahmed/Documents/Projects/IoT_Defence/Graphs/DistanceParameters/CleanTest.pkl', 'rb')
x_trainFileLevels1 = open('/home/ahmed/Documents/Projects/IoT_Defence/Graphs/DistanceParameters/CleanAvgTest.pkl', 'rb')

x_trainFileDense2 = open('/home/ahmed/Documents/Projects/IoT_Defence/Graphs/DistanceParameters/Adv.pkl', 'rb')
x_trainFileLevels2 = open('/home/ahmed/Documents/Projects/IoT_Defence/Graphs/DistanceParameters/AdvAvg.pkl', 'rb')

u = pickle._Unpickler(x_trainFileDense1)
u.encoding = 'latin1'
Clean = u.load()
u = pickle._Unpickler(x_trainFileLevels1)
u.encoding = 'latin1'
CleanAvg = u.load()

u = pickle._Unpickler(x_trainFileDense2)
u.encoding = 'latin1'
Adv = u.load()
u = pickle._Unpickler(x_trainFileLevels2)
u.encoding = 'latin1'
AdvAvg = u.load()
#
# print(len(Clean))
# print(len(CleanAvg))
# print(len(Adv))
# print(len(AdvAvg))

ApproxParameter = 1
for i in range(len(Clean)):
    Clean[i] = round(Clean[i], ApproxParameter)
for i in range(len(CleanAvg)):
    CleanAvg[i] = round(CleanAvg[i], ApproxParameter)
for i in range(len(Adv)):
    Adv[i] = round(Adv[i], ApproxParameter)
for i in range(len(AdvAvg)):
    AdvAvg[i] = round(AdvAvg[i], ApproxParameter)


CleanUnique = list(set(Clean))
CleanAvgUnique = list(set(CleanAvg))
AdvUnique = list(set(Adv))
AdvAvgUnique = list(set(AdvAvg))
CleanUnique.sort()
CleanAvgUnique.sort()
AdvUnique.sort()
AdvAvgUnique.sort()


FileToSaveAllClasses = open('/home/ahmed/Documents/Projects/IoT_Defence/AEModels/DistanceValues/AllClassesMeanSTDevDistances.pkl', 'rb')
u = pickle._Unpickler(FileToSaveAllClasses)
u.encoding = 'latin1'
MaxDistancesAllClasses = u.load()
Xthreshold = (MaxDistancesAllClasses[0]+MaxDistancesAllClasses[1])
print(Xthreshold)


CleanDist = [0] * len(CleanUnique)
CleanAvgDist = [0] * len(CleanAvgUnique)
AdvDist = [0] * len(AdvUnique)
AdvAvgDist = [0] * len(AdvAvgUnique)

for i in range(len(Clean)):
    CleanDist[CleanUnique.index(Clean[i])]+=1/len(Clean)
# print(CleanDist)
print(sum(CleanDist))

for i in range(len(CleanAvg)):
    CleanAvgDist[CleanAvgUnique.index(CleanAvg[i])]+=1/len(CleanAvg)
# print(CleanAvgDist)
print(sum(CleanAvgDist))

for i in range(len(Adv)):
    AdvDist[AdvUnique.index(Adv[i])]+=1/len(Adv)
# print(AdvDist)
print(sum(AdvDist))

for i in range(len(AdvAvg)):
    AdvAvgDist[AdvAvgUnique.index(AdvAvg[i])]+=1/len(AdvAvg)
# print(AdvAvgDist)
print(sum(AdvAvgDist))

SmoothParameter = 100

CleanUniqueSmooth = np.linspace(min(CleanUnique), max(CleanUnique), SmoothParameter)
CleanDistSmooth = spline(CleanUnique, CleanDist, CleanUniqueSmooth)

AdvUniqueSmooth = np.linspace(min(AdvUnique), max(AdvUnique), SmoothParameter)
AdvDistSmooth = spline(AdvUnique, AdvDist, AdvUniqueSmooth)

CleanAvgUniqueSmooth = np.linspace(min(CleanAvgUnique), max(CleanAvgUnique), SmoothParameter)
CleanAvgDistSmooth = spline(CleanAvgUnique, CleanAvgDist, CleanAvgUniqueSmooth)

AdvAvgUniqueSmooth = np.linspace(min(AdvAvgUnique), max(AdvAvgUnique), SmoothParameter)
AdvAvgDistSmooth = spline(AdvAvgUnique, AdvAvgDist, AdvAvgUniqueSmooth)
plt.xlabel('L2 Distance')
plt.ylabel('Frequency')
plt.grid(True)

print(Xthreshold)
plt.axvline(x=Xthreshold,linestyle='-.',color="#4f4f4f")
plt.ylim(0, max(max(CleanDistSmooth),max(AdvDistSmooth))*1.05)
plt.plot(CleanUniqueSmooth, CleanDistSmooth, label="Normal samples",linestyle='-',color="black")
plt.plot(AdvUniqueSmooth, AdvDistSmooth,label="Adversarial samples",linestyle='--',LineWidth='2',color="#303030")
plt.legend(loc='best')

plt.show()

toWrite = ""
for j in range(len(CleanUniqueSmooth)):
        toWrite += str(CleanUniqueSmooth[j])+","+str(CleanDistSmooth[j])+"\n"

f = open("/home/ahmed/Documents/Projects/IoT_Defence/TestPloting/data/REDistClean.csv","w")
f.write(toWrite)

toWrite = ""
for j in range(len(AdvUniqueSmooth)):
        toWrite += str(AdvUniqueSmooth[j])+","+str(AdvDistSmooth[j])+"\n"

f = open("/home/ahmed/Documents/Projects/IoT_Defence/TestPloting/data/REDistAdv.csv","w")
f.write(toWrite)



# plt.axvline(x=Xthreshold,linestyle='--',color="red")
# plt.ylim(0, max(max(CleanAvgDistSmooth),max(AdvAvgDistSmooth))*1.05)
# plt.plot(CleanAvgUniqueSmooth, CleanAvgDistSmooth, AdvAvgUniqueSmooth, AdvAvgDistSmooth)
# plt.show()



#### AUC ####

CleanDistAcc = []
AdvDistAcc = []
CleanAvgDistAcc = []
AdvAvgDistAcc = []


for i in range(len(CleanDist)):
    CleanDistAcc.append(sum(CleanDist[:i+1]))

for i in range(len(AdvDist)):
    AdvDistAcc.append(sum(AdvDist[:i+1]))


SmoothParameter = 100
CleanUniqueSmooth = np.linspace(min(CleanUnique), max(CleanUnique), SmoothParameter)
CleanDistSmooth = spline(CleanUnique, CleanDistAcc, CleanUniqueSmooth)

AdvUniqueSmooth = np.linspace(min(AdvUnique), max(AdvUnique), SmoothParameter)
AdvDistSmooth = spline(AdvUnique, AdvDistAcc, AdvUniqueSmooth)

iIs1 = 0
for i in range(len(AdvDistSmooth)):
    if AdvDistSmooth[i] >= 1 :
        iIs1 = 1
        AdvDistSmooth[i] = 1
    if iIs1==1 :
        AdvDistSmooth[i] = 1

iIs1 = 0
for i in range(len(CleanDistSmooth)):
    if CleanDistSmooth[i] >= 1 :
        iIs1 = 1
        CleanDistSmooth[i] = 1
    if iIs1==1 :
        CleanDistSmooth[i] = 1
    if i > 0 and CleanDistSmooth[i] < CleanDistSmooth[i-1]:
        CleanDistSmooth[i] = min(1,CleanDistSmooth[i-1]*1.001)


plt.xlabel('L2 Distance')
plt.ylabel('AUC')
plt.grid(True)

plt.axvline(x=Xthreshold,linestyle='-.',color="#4f4f4f")
plt.ylim(0, max(max(CleanDistSmooth),max(AdvDistSmooth))*1.05)
# plt.plot(CleanUnique, CleanDist, label="Normal samples")
plt.plot(CleanUniqueSmooth, CleanDistSmooth, label="Normal samples",linestyle='-',color="black")
# plt.plot(AdvUnique, AdvDist,label="Adversarial samples")
plt.plot(AdvUniqueSmooth, AdvDistSmooth,label="Adversarial samples",linestyle='--',LineWidth='2',color="#303030")
plt.legend(loc='best')

plt.show()


toWrite = ""
for j in range(len(CleanUniqueSmooth)):
        toWrite += str(CleanUniqueSmooth[j])+","+str(CleanDistSmooth[j])+"\n"

f = open("/home/ahmed/Documents/Projects/IoT_Defence/TestPloting/data/REAUCClean.csv","w")
f.write(toWrite)

toWrite = ""
for j in range(len(AdvUniqueSmooth)):
        toWrite += str(AdvUniqueSmooth[j])+","+str(AdvDistSmooth[j])+"\n"

f = open("/home/ahmed/Documents/Projects/IoT_Defence/TestPloting/data/REAUCAdv.csv","w")
f.write(toWrite)




# #### AUC ####
#
# CleanDistAcc = []
# AdvDistAcc = []
# CleanAvgDistAcc = []
# AdvAvgDistAcc = []
#
#
# for i in range(len(CleanDist)):
#     CleanDistAcc.append(sum(CleanDist[:i+1]))
#
# for i in range(len(AdvDist)):
#     AdvDistAcc.append(sum(AdvDist[:i+1]))
#
# for i in range(len(CleanAvgDist)):
#     CleanAvgDistAcc.append(sum(CleanAvgDist[:i+1]))
#
# for i in range(len(AdvAvgDist)):
#     AdvAvgDistAcc.append(sum(AdvAvgDist[:i+1]))
#
#
# SmoothParameter = 100
#
# CleanUniqueSmooth = np.linspace(min(CleanUnique), max(CleanUnique), SmoothParameter)
# CleanDistSmooth = spline(CleanUnique, CleanDistAcc, CleanUniqueSmooth)
#
# AdvUniqueSmooth = np.linspace(min(AdvUnique), max(AdvUnique), SmoothParameter)
# AdvDistSmooth = spline(AdvUnique, AdvDistAcc, AdvUniqueSmooth)
#
# CleanAvgUniqueSmooth = np.linspace(min(CleanAvgUnique), max(CleanAvgUnique), SmoothParameter)
# CleanAvgDistSmooth = spline(CleanAvgUnique, CleanAvgDistAcc, CleanAvgUniqueSmooth)
#
# AdvAvgUniqueSmooth = np.linspace(min(AdvAvgUnique), max(AdvAvgUnique), SmoothParameter)
# AdvAvgDistSmooth = spline(AdvAvgUnique, AdvAvgDistAcc, AdvAvgUniqueSmooth)
#
# plt.axvline(x=Xthreshold,linestyle='--',color="red")
# plt.ylim(0, max(max(CleanDistSmooth),max(AdvDistSmooth))*1.05)
# plt.plot(CleanUniqueSmooth, CleanDistSmooth, AdvUniqueSmooth, AdvDistSmooth)
# plt.show()

#
# plt.axvline(x=Xthreshold,linestyle='--',color="red")
# plt.ylim(0, max(max(CleanAvgDistSmooth),max(AdvAvgDistSmooth))*1.05)
# plt.plot(CleanAvgUniqueSmooth, CleanAvgDistSmooth, AdvAvgUniqueSmooth, AdvAvgDistSmooth)
# plt.show()
