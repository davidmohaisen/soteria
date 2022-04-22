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
baseDirectory = "/home/ahmed/Documents/Projects/IoT_Defence/Attack/Raw/"
Families = ["benign","gafgyt","mirai","tsunami"]
FamiliesRank = [0,1,2,3]
Choices = ["benignS","benignM","benignL","gafgytS","gafgytM","gafgytL","miraiS","miraiM","miraiL","tsunamiS","tsunamiM","tsunamiL"]
ChoicesRank = [0,0,0,1,1,1,2,2,2,3,3,3]



def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=50, source_samples=10,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):

    # Object used to keep track of (and return) key accuracies

    # ToHandle = "dense"
    # Features = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/Dense_Features.pkl', 'rb')
    # x0 = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XDense.pkl', 'wb')
    # x1 = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XGroundTruthDense.pkl', 'wb')
    # x2 = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialLabelDense.pkl', 'wb')
    # x3 = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialSizeUsedDense.pkl', 'wb')

    ToHandle = "levels"
    Features = open('/home/ahmed/Documents/Projects/IoT_Defence/pkl/Levels_Features.pkl', 'rb')
    x0 = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XLevels.pkl', 'wb')
    x1 = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XGroundTruthLevels.pkl', 'wb')
    x2 = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialLabelLevels.pkl', 'wb')
    x3 = open('/home/ahmed/Documents/Projects/IoT_Defence/Attack/Processed/XAdversarialSizeUsedLevels.pkl', 'wb')



    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    Attack_X = []
    Attack_Ground_Truth = []
    Attack_Adversarial_Label = []
    Attack_Adversarial_Size_Used = []
    for a in range(len(Families)):
        for b in range(len(Choices)):
            if b >= 3*a and b <= 3*a+2 :
                continue

            toOpen = baseDirectory + Families[a] + "/" + Choices[b] + "/" +ToHandle + ".pkl"
            RWFile = open(toOpen, 'rb')
            u = pickle._Unpickler(RWFile)
            u.encoding = 'latin1'
            RW = u.load()

            for i in range(len(RW)):
                Attack_X.append(RW[i])
                Attack_Ground_Truth.append(FamiliesRank[a])
                Attack_Adversarial_Label.append(ChoicesRank[b])
                Attack_Adversarial_Size_Used.append(Choices[b])

    print(len(Attack_X))


    u = pickle._Unpickler(Features)
    u.encoding = 'latin1'
    Features = u.load()

    vectorizer = TfidfVectorizer(ngram_range=(2,4), max_features=500, vocabulary=Features)
    X = vectorizer.fit_transform(Attack_X)
    X = X.toarray()
    print("X:",X.shape)

    pickle.dump(X, x0)
    pickle.dump(Attack_Ground_Truth, x1)
    pickle.dump(Attack_Adversarial_Label, x2)
    pickle.dump(Attack_Adversarial_Size_Used, x3)
    exit()



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
