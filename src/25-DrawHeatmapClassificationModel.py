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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
FLAGS = flags.FLAGS

LEARNING_RATE = .001
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
    x_test = np.loadtxt('/home/ahmed/Documents/sdn/dataVectors/test8VectorUpdated.txt')
    y_test = np.loadtxt('/home/ahmed/Documents/sdn/dataVectors/test8LabelsUpdated.txt')
    y_testBefore = y_test
    x_test = x_test.reshape((34717,68,1))
    y_test = keras.utils.to_categorical(y_test, 8)

    ###########################################################################################
    # Load the model using TensorFlow
    ###########################################################################################
    # Obtain Image Parameters
    img_rows, img_cols = x_test.shape[1:3]
    nb_classes = y_test.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 68

    # Define TF model graph
    model = ModelBasicCNN(nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    saver = tf.train.Saver()
    saver.restore(sess, "/home/ahmed/Documents/sdn/model/UpdatedModel8.ckpt")
    print("Model restored.")
    ###########################################################################################
    predicts = model_argmax(sess,x,preds,x_test)
    confusion = []
    for i in range(8):
        confusion.append([])
        for j in range(8):
            confusion[i].append(0)

    for i in range(len(y_testBefore)):
        confusion[int(predicts[i])][int(y_testBefore[i])]+=1

    co = confusion
    max = [0]*8
    # max = [0]*2
    for i in range(len(co)):
        for j in range(len(co[i])):
            max[j] += co[i][j]
    for i in range(len(co)):
        for j in range(len(co[i])):
            co[i][j] = co[i][j]/max[j]
            co[i][j] = round(co[i][j],3)


    co = np.asarray(co)
    df = pd.DataFrame(co, columns=["C0","C1","C2","C3","C4","C5","C6","C7"],index=["C0","C1","C2","C3","C4","C5","C6","C7"])
    # df = pd.DataFrame(co, columns=["C0","C1"],index=["C0","C1"])
    sns.set(font_scale=0.925)
    ax = sns.heatmap(df,annot=True,cmap="Blues",fmt='g')
    plt.savefig("/home/ahmed/Documents/sdn/HMaps/ModelClassification.pdf")
    plt.clf()
    plt.close()

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
    flags.DEFINE_integer('nb_epochs', 100, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
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
