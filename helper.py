# TODO: Prepare the data function

import numpy as np

import mat4py

import matplotlib.pylab as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pylab as plt
import itertools

import random


###################################################################
###################### Undersampling function #####################
###################################################################
def balance_labels(labels):
    new_indices_1 = [i for i, x in enumerate(labels) if x == 1]
    len_1 = len(new_indices_1)
    new_indices_0 = [i for i, x in enumerate(labels) if x == 0]
    len_0 = len(new_indices_0)

    if len_1 > len_0:
        n_sample_to_delete = len_1 - len_0
        list_of_random_idx = random.sample(new_indices_1, n_sample_to_delete)

    elif len_0 >= len_1:
        n_sample_to_delete = len_0 - len_1
        list_of_random_idx = random.sample(new_indices_0, n_sample_to_delete)

    mask = np.ones(len(labels), dtype=bool)
    mask[list_of_random_idx] = False
    return mask


###################################################################
##################### Plot Confusion Matrix #######################
###################################################################
def plot_confusion_matrix(cm, classes, filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, bbox_inches='tight')


###################################################################
######################### Plot PR Curve ###########################
###################################################################
def plot_precision_recall_curve(y_test, y_score,
                                filename):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig(filename, bbox_inches='tight')


###################################################################
################### Plot Loss-Accuracy Curve ######################
###################################################################
def plot_loss_acc(history, filename):
    fig = plt.figure(figsize=(20, 10))
    subfig = fig.add_subplot(122)
    subfig.plot(history.history['acc'], label="training")
    if history.history['val_acc'] is not None:
        subfig.plot(history.history['val_acc'], label="testing")
    subfig.set_title('Model Accuracy')
    subfig.set_xlabel('Epoch')
    subfig.legend(loc='upper left')
    plt.grid()
    subfig = fig.add_subplot(121)
    subfig.plot(history.history['loss'], label="training")
    if history.history['val_loss'] is not None:
        subfig.plot(history.history['val_loss'], label="testing")
    subfig.set_title('Model Loss')
    subfig.set_xlabel('Epoch')
    subfig.legend(loc='upper left')
    plt.grid()
    fig.savefig(filename, bbox_inches='tight')


###################################################################
##################### Save metrics into .csv ######################
###################################################################
def metrics_to_csv(metrics, filename):
    np.savetxt(filename, metrics, delimiter=',')


###################################################################
######################## Data Preparation #########################
###################################################################

def load_single_file(filename):
    temp = mat4py.loadmat(filename)
    return temp['data']


def load_data(filename_list):
    data = [None] * len(filename_list)
    for i, f_ in enumerate(filename_list):
        data[i] = load_single_file(filename=f_)
    return data


def simplify_data_per_run(run_data):
    # TODO: Remove artifacts from dataset? - To ask to Tibor

    # simple_X = zeros(len(run_data['trial'])-sum(run_data['artifact']))
    # simple_y = zeros(len(run_data['trial'])-sum(run_data['artifact']))

    simple_X = [None] * len(run_data['trial'])
    simple_y = [None] * len(run_data['trial'])

    for i, t_ in enumerate(run_data['trial']):
        # trials are not 'int' but list of one --> take t_[0] gives the correct value
        simple_X[i] = run_data['X'][t_[0] - 125:t_[0] + 1000]
        simple_y[i] = run_data['y'][i]

    return simple_X, simple_y


def simplify_data_per_user(user_data):
    simple_user_X = [None] * 6
    simple_user_y = [None] * 6

    for i, run in enumerate(user_data[3:8]):
        simple_user_X[i], simple_user_y[i] = simplify_data_per_run(run_data=run)

    return simple_user_X, simple_user_y


def load_simplify_per_user(filename):
    temp = simplify_data_per_user(load_single_file(filename))

    return temp


def load_simplify_data(filename_list):
    data = [None] * len(filename_list)

    for i, f_ in enumerate(filename_list):
        data[i] = load_simplify_per_user(f_)

    return data

list = ['Data/A01T.mat', 'Data/A01T.mat']

load_simplify_data(list)