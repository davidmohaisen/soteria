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

    x_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/DSSeparatedRaw/x_train_Dense_Raw.pkl', 'rb')
    y_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/DSSeparatedRaw/y_train_Dense_Raw.pkl', 'rb')

    x_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/DSSeparatedRaw/x_train_Levels_Raw.pkl', 'rb')
    y_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/DSSeparatedRaw/y_train_Levels_Raw.pkl', 'rb')

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


    ###### Test #####

    FeaturesDense = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/Dense_Features.pkl', 'rb')
    FeaturesLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/Levels_Features.pkl', 'rb')
    u = pickle._Unpickler(FeaturesDense)
    u.encoding = 'latin1'
    FeaturesAllDense = u.load()

    u = pickle._Unpickler(FeaturesLevels)
    u.encoding = 'latin1'
    FeaturesAllLevels = u.load()

    class0Dense = []
    class1Dense = []
    class2Dense = []
    class3Dense = []

    class0Levels = []
    class1Levels = []
    class2Levels = []
    class3Levels = []

    for i in range(len(x_trainDense)):
        if y_trainDense[i] == 0 :
            class0Dense.append(x_trainDense[i])
        if y_trainDense[i] == 1 :
            class1Dense.append(x_trainDense[i])
        if y_trainDense[i] == 2 :
            class2Dense.append(x_trainDense[i])
        if y_trainDense[i] == 3 :
            class3Dense.append(x_trainDense[i])

    for i in range(len(x_trainLevels)):
        if y_trainLevels[i] == 0 :
            class0Levels.append(x_trainLevels[i])
        if y_trainLevels[i] == 1 :
            class1Levels.append(x_trainLevels[i])
        if y_trainLevels[i] == 2 :
            class2Levels.append(x_trainLevels[i])
        if y_trainLevels[i] == 3 :
            class3Levels.append(x_trainLevels[i])


    vectorizer0Dense = TfidfVectorizer(ngram_range=(2,4), max_features=500)
    vectorizer1Dense = TfidfVectorizer(ngram_range=(2,4), max_features=500)
    vectorizer2Dense = TfidfVectorizer(ngram_range=(2,4), max_features=500)
    vectorizer3Dense = TfidfVectorizer(ngram_range=(2,4), max_features=500)

    vectorizer0Levels = TfidfVectorizer(ngram_range=(2,4), max_features=500)
    vectorizer1Levels = TfidfVectorizer(ngram_range=(2,4), max_features=500)
    vectorizer2Levels = TfidfVectorizer(ngram_range=(2,4), max_features=500)
    vectorizer3Levels = TfidfVectorizer(ngram_range=(2,4), max_features=500)


    X0Dense = vectorizer0Dense.fit_transform(class0Dense)
    X1Dense = vectorizer1Dense.fit_transform(class1Dense)
    X2Dense = vectorizer2Dense.fit_transform(class2Dense)
    X3Dense = vectorizer3Dense.fit_transform(class3Dense)
    FeaturesAllDense = set(list(FeaturesAllDense))
    vectorizer0Dense = set(list(vectorizer0Dense.vocabulary_))
    vectorizer1Dense = set(list(vectorizer1Dense.vocabulary_))
    vectorizer2Dense = set(list(vectorizer2Dense.vocabulary_))
    vectorizer3Dense = set(list(vectorizer3Dense.vocabulary_))


    X0Levels = vectorizer0Levels.fit_transform(class0Levels)
    X1Levels = vectorizer1Levels.fit_transform(class1Levels)
    X2Levels = vectorizer2Levels.fit_transform(class2Levels)
    X3Levels = vectorizer3Levels.fit_transform(class3Levels)
    FeaturesAllLevels = set(list(FeaturesAllLevels))
    vectorizer0Levels = set(list(vectorizer0Levels.vocabulary_))
    vectorizer1Levels = set(list(vectorizer1Levels.vocabulary_))
    vectorizer2Levels = set(list(vectorizer2Levels.vocabulary_))
    vectorizer3Levels = set(list(vectorizer3Levels.vocabulary_))

    vectorizer0Dense = FeaturesAllDense & vectorizer0Dense
    vectorizer1Dense = FeaturesAllDense & vectorizer1Dense
    vectorizer2Dense = FeaturesAllDense & vectorizer2Dense
    vectorizer3Dense = FeaturesAllDense & vectorizer3Dense

    vectorizer0Levels = FeaturesAllLevels & vectorizer0Levels
    vectorizer1Levels = FeaturesAllLevels & vectorizer1Levels
    vectorizer2Levels = FeaturesAllLevels & vectorizer2Levels
    vectorizer3Levels = FeaturesAllLevels & vectorizer3Levels


    # print("With Total : Dense")
    # print("Benign :",len(vectorizer0Dense))
    # print("Gafgyt :",len(vectorizer1Dense))
    # print("Mirai :",len(vectorizer2Dense))
    # print("Tsunami :",len(vectorizer3Dense))
    #
    # print("With Total : Levels")
    # print("Benign :",len(vectorizer0Levels))
    # print("Gafgyt :",len(vectorizer1Levels))
    # print("Mirai :",len(vectorizer2Levels))
    # print("Tsunami :",len(vectorizer3Levels))


    FeaturesAllDense = list(FeaturesAllDense)
    FeaturesAllLevels = list(FeaturesAllLevels)
    vectorizer0Dense = list(vectorizer0Dense)
    vectorizer1Dense = list(vectorizer1Dense)
    vectorizer2Dense = list(vectorizer2Dense)
    vectorizer3Dense = list(vectorizer3Dense)
    vectorizer0Levels = list(vectorizer0Levels)
    vectorizer1Levels = list(vectorizer1Levels)
    vectorizer2Levels = list(vectorizer2Levels)
    vectorizer3Levels = list(vectorizer3Levels)

    # print(type(FeaturesAllDense))
    # print(type(FeaturesAllLevels))

    DenseSelectedFeatures = [vectorizer0Dense,vectorizer1Dense,vectorizer2Dense,vectorizer3Dense]
    # print(DenseSelectedFeatures)
    # print(len(DenseSelectedFeatures))
    # print(len(DenseSelectedFeatures[0]))

    LevelsSelectedFeatures = [vectorizer0Levels,vectorizer1Levels,vectorizer2Levels,vectorizer3Levels]
    # print(LevelsSelectedFeatures)
    # print(len(LevelsSelectedFeatures))
    # print(len(LevelsSelectedFeatures[0]))

    DenseIndexes = [[],[],[],[]]
    LevelsIndexes = [[],[],[],[]]

    # print(FeaturesAllDense)
    # print(len(FeaturesAllDense))

    for i in range(len(DenseSelectedFeatures)):
        for j in range(len(DenseSelectedFeatures[i])):
            DenseIndexes[i].append(FeaturesAllDense.index(DenseSelectedFeatures[i][j]))


    for i in range(len(LevelsSelectedFeatures)):
        for j in range(len(LevelsSelectedFeatures[i])):
            LevelsIndexes[i].append(FeaturesAllLevels.index(LevelsSelectedFeatures[i][j]))

    for i in range(len(DenseIndexes)):
        print(len(DenseIndexes[i]))

    for i in range(len(LevelsIndexes)):
        print(len(LevelsIndexes[i]))


    x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/DenseIndexes.pkl', 'wb')
    pickle.dump(DenseIndexes, x_trainFile)
    x_trainFile = open('/home/ahmed/Documents/Projects/IoT_Defence/ScoreParameters/LevelsIndexes.pkl', 'wb')
    pickle.dump(LevelsIndexes, x_trainFile)

    # x_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_train_Dense.pkl', 'rb')
    # y_trainFileDense = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/y_train_Dense.pkl', 'rb')
    #
    # x_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/x_train_Levels.pkl', 'rb')
    # y_trainFileLevels = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/y_train_Levels.pkl', 'rb')
    #
    # u = pickle._Unpickler(x_trainFileDense)
    # u.encoding = 'latin1'
    # x_trainDense = u.load()
    # u = pickle._Unpickler(y_trainFileDense)
    # u.encoding = 'latin1'
    # y_trainDense = u.load()
    #
    # u = pickle._Unpickler(x_trainFileLevels)
    # u.encoding = 'latin1'
    # x_trainLevels = u.load()
    # u = pickle._Unpickler(y_trainFileLevels)
    # u.encoding = 'latin1'
    # y_trainLevels = u.load()
    #
    # scores = [[]]*4
    # scores[0] = [[]]*1000
    # scores[1] = [[]]*1000
    # scores[2] = [[]]*1000
    # scores[3] = [[]]*1000


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
