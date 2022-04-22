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



FLAGS = flags.FLAGS

LEARNING_RATE = .002
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 1000



def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=50, source_samples=10,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

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

    # Adv
    x_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XDense.pkl', 'rb')
    y_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialLabelDense.pkl', 'rb')

    x_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XLevels.pkl', 'rb')
    y_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialLabelLevels.pkl', 'rb')


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

    ##### Get ScoreValues List #####
    # DenseFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/DenseIndexes.pkl', 'rb')
    # LevelsFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/LevelsIndexes.pkl', 'rb')
    #
    # u = pickle._Unpickler(DenseFile)
    # u.encoding = 'latin1'
    # DenseIndexes = u.load()
    # u = pickle._Unpickler(LevelsFile)
    # u.encoding = 'latin1'
    # LevelsIndexes = u.load()
    # scores = []
    # for i in range(4):
    #     scores.append([])
    #     for j in range(1000):
    #         scores[i].append([])
    # print((np.asarray(scores)).shape)
    #
    # for i in range(len(x_trainDense)):
    #     for j in range(len(x_trainDense[i])):
    #         if j in DenseIndexes[y_trainDense[i]]:
    #             scores[y_trainDense[i]][j].append(x_trainDense[i][j])
    #
    #         if j in LevelsIndexes[y_trainLevels[i]]:
    #             scores[y_trainLevels[i]][500+j].append(x_trainLevels[i][j])
    #
    #
    # Values = []x_trainDense
    # for i in range(4):
    #     Values.append([])
    #     for j in range(1000):
    #         Values[i].append([])
    #         if len(scores[i][j]) != 0:
    #             Values[i][j].append(mean(scores[i][j]))
    #             Values[i][j].append(stdev(scores[i][j]))
    #
    # for i in range(4):
    #     for j in range(1000):
    #         AccDistance = 0
    #         NumSamples = 0
    #         for k in range(len(scores[i][j])):
    #             if scores[i][j][k] < Values[i][j][0]-Values[i][j][1] or scores[i][j][k] > Values[i][j][0]+Values[i][j][1] :
    #                 if scores[i][j][k] < Values[i][j][0]-Values[i][j][1]:
    #                     AccDistance += (Values[i][j][0]-Values[i][j][1])-scores[i][j][k]
    #                 elif scores[i][j][k] > Values[i][j][0]+Values[i][j][1] :
    #                     AccDistance += scores[i][j][k] - (Values[i][j][0]+Values[i][j][1])
    #                 NumSamples += 1
    #         if len(scores[i][j]) != 0 :
    #             if NumSamples != 0 :
    #                 Values[i][j].append(AccDistance/NumSamples)
    #             else :
    #                 Values[i][j].append(AccDistance)
    #
    #
    # x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/ScoreValuesDenseLevels.pkl', 'wb')
    # pickle.dump(Values, x_trainFile)
    #
    # print(Values)
    # print((np.asarray(Values)).shape)

    ##### Load ScoreValues List #####

    ScoreValuesArray = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/ScoreValuesDenseLevels.pkl', 'rb')

    u = pickle._Unpickler(ScoreValuesArray)
    u.encoding = 'latin1'
    ScoreValuesArray = u.load()


    ##### Generate Distances #####
    # NormalDenseDistance = [[],[],[],[]]
    # NormalLevelsDistance = [[],[],[],[]]
    # NumOfZeroDense = [[],[],[],[]]
    # NumOfZeroLevels = [[],[],[],[]]
    #
    #
    # for i in range(int(len(x_trainDense)/10)):
    #     oneSampleDense = []
    #     oneSampleLevels = []
    #     zerosDense = 0
    #     zerosLevels = 0
    #     for m in range(10):
    #         for j in range(len(x_trainDense[(i*10)+m])):
    #             if len(ScoreValuesArray[y_trainDense[(i*10)+m]][j]) != 0:
    #                 if x_trainDense[(i*10)+m][j] == 0 :
    #                     zerosDense+= 1
    #                 distanceDense = (ScoreValuesArray[y_trainDense[(i*10)+m]][j][0] - x_trainDense[(i*10)+m][j])**2
    #                 oneSampleDense.append(distanceDense)
    #             if len(ScoreValuesArray[y_trainLevels[(i*10)+m]][500+j]) != 0:
    #                 if x_trainLevels[(i*10)+m][j] == 0 :
    #                     zerosLevels+= 1
    #                 distanceLevels = (ScoreValuesArray[y_trainLevels[(i*10)+m]][500+j][0] - x_trainLevels[(i*10)+m][j])**2
    #                 oneSampleLevels.append(distanceLevels)
    #
    #     NormalDenseDistance[y_trainLevels[(i*10)]].append((sum(oneSampleDense)/10)**(1/2))
    #     NormalLevelsDistance[y_trainLevels[(i*10)]].append((sum(oneSampleLevels)/10)**(1/2))
    #     NumOfZeroDense[y_trainLevels[(i*10)]].append(zerosDense/10)
    #     NumOfZeroLevels[y_trainLevels[(i*10)]].append(zerosLevels/10)
    #
    # ArrayToSaveDense = [[],[],[],[]]
    # ArrayToSaveLevels = [[],[],[],[]]
    # for i in range(len(NormalDenseDistance)):
    #     ArrayToSaveDense[i].append(max(NormalDenseDistance[i]))
    #     ArrayToSaveLevels[i].append(max(NormalLevelsDistance[i]))
    #     ArrayToSaveDense[i].append((sum(NormalDenseDistance[i]))/len(NormalDenseDistance[i]))
    #     ArrayToSaveLevels[i].append((sum(NormalLevelsDistance[i]))/len(NormalLevelsDistance[i]))
    #     ArrayToSaveDense[i].append(min(NormalDenseDistance[i]))
    #     ArrayToSaveLevels[i].append(min(NormalLevelsDistance[i]))
    #
    #     ArrayToSaveDense[i].append(max(NumOfZeroDense[i]))
    #     ArrayToSaveLevels[i].append(max(NumOfZeroLevels[i]))
    #     ArrayToSaveDense[i].append((sum(NumOfZeroDense[i]))/len(NumOfZeroDense[i]))
    #     ArrayToSaveLevels[i].append((sum(NumOfZeroLevels[i]))/len(NumOfZeroLevels[i]))
    #     ArrayToSaveDense[i].append(min(NumOfZeroDense[i]))
    #     ArrayToSaveLevels[i].append(min(NumOfZeroLevels[i]))
    #
    #
    # print(ArrayToSaveDense)
    # print(ArrayToSaveLevels)
    # denseValues = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/L2DistanceMaxAvgDense.pkl', 'wb')
    # LevelsValues = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/L2DistanceMaxAvgLevels.pkl', 'wb')
    # pickle.dump(ArrayToSaveDense, denseValues)
    # pickle.dump(ArrayToSaveLevels, LevelsValues)
    # exit()

    ##### Load MaxAverage Lists #####
    denseValues = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/L2DistanceMaxAvgDense.pkl', 'rb')
    LevelsValues = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/L2DistanceMaxAvgLevels.pkl', 'rb')
    u = pickle._Unpickler(denseValues)
    u.encoding = 'latin1'
    denseValues = u.load()
    u = pickle._Unpickler(LevelsValues)
    u.encoding = 'latin1'
    LevelsValues = u.load()



    ##### Start Predicting #####

    #### L2 Base detection of AE ####
    # x_trainDense = x_trainDense[0:100]



    totalScoresPerClass = [0,0,0,0]
    AllPerClass = [0,0,0,0]

    for i in range(int(len(x_trainDense)/10)):
        DistancesDense = []
        DistancesLevels = []
        zerosLevels = []
        zerosDense = []
        for m in range(10):
            oneSampleDense = []
            oneSampleLevels = []
            LevelZero = 0
            DenseZero = 0
            for j in range(len(x_trainDense[(i*10)+m])):
                if len(ScoreValuesArray[y_trainDense[(i*10)+m]][j]) != 0:
                    if x_trainDense[(i*10)+m][j] < 0.25 * ScoreValuesArray[y_trainDense[(i*10)+m]][j][0] :
                        if x_trainDense[(i*10)+m][j] < 0.1 * ScoreValuesArray[y_trainDense[(i*10)+m]][j][0] :
                            DenseZero += 2
                        else :
                            DenseZero += 1
                    distanceDense = (ScoreValuesArray[y_trainDense[(i*10)+m]][j][0] - x_trainDense[(i*10)+m][j])**2
                    oneSampleDense.append(distanceDense)

                if len(ScoreValuesArray[y_trainLevels[(i*10)+m]][500+j]) != 0:
                    if x_trainLevels[(i*10)+m][j] < 0.25 * ScoreValuesArray[y_trainLevels[(i*10)+m]][500+j][0] :
                        if x_trainLevels[(i*10)+m][j] < 0.1 * ScoreValuesArray[y_trainLevels[(i*10)+m]][500+j][0] :
                            LevelZero += 2
                        else :
                            LevelZero += 1
                    distanceLevels = (ScoreValuesArray[y_trainLevels[(i*10)+m]][500+j][0] - x_trainLevels[(i*10)+m][j])**2
                    oneSampleLevels.append(distanceLevels)
            DistancesDense.append((sum(oneSampleDense))**(1/2))
            DistancesLevels.append((sum(oneSampleLevels))**(1/2))
            zerosDense.append(DenseZero)
            zerosLevels.append(LevelZero)

        # print(zerosDense)
        score = 0
        Zscore = 0
        FactorSigma = 1.0 #CONFIG
        ZeroFactorSigma = 0.25 #CONFIG
        for n in range(len(DistancesDense)):

            if denseValues[y_trainDense[(i*10)]][0] < DistancesDense[n] :
                score += 2
            elif denseValues[y_trainDense[(i*10)]][1]*(FactorSigma) < DistancesDense[n] :
                score += 1
            elif denseValues[y_trainDense[(i*10)]][2] > DistancesDense[n] :
                score += 2

            if LevelsValues[y_trainLevels[(i*10)]][0] < DistancesLevels[n] :
                score += 2
            elif LevelsValues[y_trainLevels[(i*10)]][1]*(FactorSigma) < DistancesLevels[n] :
                score += 1
            elif LevelsValues[y_trainLevels[(i*10)]][2] > DistancesLevels[n] :
                score += 2

            if denseValues[y_trainDense[(i*10)]][4]+(ZeroFactorSigma*(denseValues[y_trainDense[(i*10)]][3]-denseValues[y_trainDense[(i*10)]][4])) < zerosDense[n] :
                Zscore += 1
            if LevelsValues[y_trainLevels[(i*10)]][4]+(ZeroFactorSigma*(LevelsValues[y_trainDense[(i*10)]][3]-(LevelsValues[y_trainDense[(i*10)]][4]))) < zerosLevels[n] :
                Zscore += 1


        # print(DistancesDense,Zscore, score)
        maxScore = 40
        zScoreMax = 35
        AllPerClass[y_trainLevels[(i*10)]]+= 1
        if score/maxScore >=0.70 or (Zscore/zScoreMax >= 0.50 and score/maxScore >=0.5):
            totalScoresPerClass[y_trainLevels[(i*10)]]+= 1

    print(totalScoresPerClass)
    print(AllPerClass)

    # totalScoresPerClass = [0,0,0,0]
    # # x_trainDense = x_trainDense[:100]
    # for i in range(int(len(x_trainDense)/10)):
    #     All = [0,0,0,0]
    #     Distances = [0,0,0,0]
    #     for m in range(10):
    #         for j in range(len(x_trainDense[(i*10)+m])):
    #             for k in range(len(ScoreValuesArray)):
    #                 if (len(ScoreValuesArray[k][j]) != 0 and len(ScoreValuesArray[y_trainDense[(i*10)+m]][j]) == 0) or  (k == y_trainDense[(i*10)+m] and len(ScoreValuesArray[y_trainDense[(i*10)+m]][j]) != 0):
    #                     All[k] += 1
    #
    #                     if x_trainDense[(i*10)+m][j] < ScoreValuesArray[k][j][0]-ScoreValuesArray[k][j][1]:
    #                         Distances[k] += ScoreValuesArray[k][j][0]-ScoreValuesArray[k][j][1] - x_trainDense[(i*10)+m][j]
    #                     else :
    #                         Distances[k] += x_trainDense[(i*10)+m][j] - ScoreValuesArray[k][j][0]+ScoreValuesArray[k][j][1]
    #
    #
    #                     # if x_trainDense[(i*10)+m][j] != 0:
    #                     #     if x_trainDense[(i*10)+m][j] < ScoreValuesArray[k][j][0]-ScoreValuesArray[k][j][1] or x_trainDense[(i*10)+m][j] > ScoreValuesArray[k][j][0]+ScoreValuesArray[k][j][1] :
    #                     #         Distances[k] +=1
    #                     # else:
    #                     #     Distances[k] +=1
    #
    #                 if len(ScoreValuesArray[k][500+j]) != 0 and len(ScoreValuesArray[y_trainDense[(i*10)+m]][500+j]) == 0 or  (k == y_trainDense[(i*10)+m] and len(ScoreValuesArray[y_trainDense[(i*10)+m]][500+j]) != 0):
    #                     All[k] += 1
    #
    #                     if x_trainLevels[(i*10)+m][j] < ScoreValuesArray[k][500+j][0]-ScoreValuesArray[k][500+j][1]:
    #                         Distances[k] += ScoreValuesArray[k][500+j][0]-ScoreValuesArray[k][500+j][1] - x_trainLevels[(i*10)+m][j]
    #                     else :
    #                         Distances[k] += x_trainLevels[(i*10)+m][j] - ScoreValuesArray[k][500+j][0]+ScoreValuesArray[k][500+j][1]
    #
    #                     # if x_trainLevels[(i*10)+m][j] != 0 :
    #                     #     if x_trainLevels[(i*10)+m][j] < ScoreValuesArray[k][500+j][0]-ScoreValuesArray[k][500+j][1] or x_trainLevels[(i*10)+m][j] > ScoreValuesArray[k][500+j][0]+ScoreValuesArray[k][500+j][1] :
    #                     #         Distances[k] +=1
    #                     # else:
    #                     #     Distances[k] +=1
    #
    #
    #
    #     minimum = 0
    #     for j in range(4):
    #         if All[j]!= 0 :
    #             Distances[j] = Distances[j] / All[j]
    #
    #     # print(Distances)
    #
    #     minimum = min(Distances)
    #     if minimum != Distances[y_trainDense[i*10]] or minimum == 1:
    #         totalScoresPerClass[y_trainDense[i*10]] += 1
    #         # print("found!")
    #
    #
    #
    #
    # print(totalScoresPerClass)



def main(argv=None):
    mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
                      attack_iterations=FLAGS.attack_iterations,
                      model_path=FLAGS.model_path,
                      targeted=FLAGS.targeted)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
    flags.DEFINE_integer('source_samples', 1091, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')

    tf.app.run()
