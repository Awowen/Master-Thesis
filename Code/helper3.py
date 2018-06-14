# TODO: Prepare the data function
# TODO: Add exceptions

import numpy as np

import mat4py

import matplotlib.pylab as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy import signal
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
    plt.savefig('{}_confusion.png'.format(filename), bbox_inches='tight')


###################################################################
######################### Plot PR Curve ###########################
###################################################################
def plot_precision_recall_curve(y_test, y_score,
                                filename):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(4):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    precision['micro'], recall['micro'], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())
    average_precision['micro'] = average_precision_score(y_test, y_score,
                                                         average='micro')

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=.2, where='post')
    plt.fill_between(recall['micro'], precision['micro'], step='post', alpha=.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([.0, 1.05])
    plt.xlim([.0, 1.0])
    plt.title('Average precision score, micro averaged over all classes: AP={0:0.2f}'.format(average_precision['micro']))
    plt.savefig('{}_precision_recall.png'.format(filename), bbox_incches='tight')
    return average_precision['micro']


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
    fig.savefig('{}_loss_acc.png'.format(filename), bbox_inches='tight')


###################################################################
##################### Save metrics into .csv ######################
###################################################################
def metrics_to_csv(metrics, filename):
    np.savetxt('{}.csv'.format(filename), metrics, delimiter=',')


def prediction_vector(y_prob):
    y_pred = y_prob.argmax(axis=-1)
    return y_pred


def get_metrics_and_plots(labels, y_score, history, filename, class_names):
    values = np.zeros(4)
    # Plot the confusion matrix

    y_pred = prediction_vector(y_score)
    testing_labels = labels
    labels = labels.argmax(axis=-1)
    cnf_matrix = confusion_matrix(labels, y_pred)
    plot_confusion_matrix(cnf_matrix, class_names, filename)
    # Plot the precision recall curve
    plot_precision_recall_curve(testing_labels, y_score, filename)
    # Plot the loss and accuracy
    plot_loss_acc(history, filename)

    # Accuracy
    values[0] = accuracy_score(labels, y_pred)
    # Precision
    values[1] = precision_score(labels, y_pred, average='micro')
    # Recall
    values[2] = recall_score(labels, y_pred, average='micro')
    # f1 score
    values[3] = f1_score(labels, y_pred, average='micro')

    return values


###################################################################
##################### Data Preparation function ###################
###################################################################

def load_single_file(filename):
    temp = mat4py.loadmat(filename)
    return temp['data']


def load_data(filename_list):
    data = [None] * len(filename_list)
    for i, f_ in enumerate(filename_list):
        data[i] = load_single_file(filename=f_)
    return data


def simplify_data_per_run(run_data, highpass, eog, chs):

    # simple_X = zeros(len(run_data['trial'])-sum(run_data['artifact']))
    # simple_y = zeros(len(run_data['trial'])-sum(run_data['artifact']))

    simple_X = [None] * len(run_data['trial'])
    simple_y = [None] * len(run_data['trial'])
    if highpass:
        run_data['X'] = butter_highpass_filter(run_data['X'], 4)
    for i, t_ in enumerate(run_data['trial']):
        # trials are not 'int' but list of one --> take t_[0] gives the correct value
        # Same for labels
        if eog:
            simple_X[i] = remove_eog(np.asanyarray(run_data['X'][t_[0] + 125:t_[0] + 1250]))
            if chs:
                simple_X[i] = simple_X[i][:, chs]
        if not eog:
            simple_X[i] = run_data['X'][t_[0] + 125:t_[0] + 1250]
            if chs:
                chs = chs
                simple_X[i] = simple_X[i][:, chs]

        simple_y[i] = run_data['y'][i][0]

    return simple_X, simple_y


def simplify_data_per_user(user_data, highpass, eog, chs):
    simple_user_X = [None] * len(user_data)
    simple_user_y = [None] * len(user_data)

    for i, run in enumerate(user_data):
        simple_user_X[i], simple_user_y[i] = simplify_data_per_run(run, highpass, eog, chs)

    return simple_user_X, simple_user_y


def load_simplify_per_user(filename, highpass, eog, chs):
    temp = simplify_data_per_user(load_single_file(filename), highpass, eog, chs)
    return temp


def load_simplify_data(filename_list, highpass=False, eog=False, chs=[]):
    data = [None] * len(filename_list)

    for i, f_ in enumerate(filename_list):
        data[i] = load_simplify_per_user(f_, highpass, eog, chs)

    return data


###################################################################
##################### Data Preparation function ###################
###################################################################
def prepare_data_standard_from_list_split(data, size):
    big_X_train = [None] * len(data)
    big_X_test = [None] * len(data)
    big_y_train = [None] * len(data)
    big_y_test = [None] * len(data)

    for i, u_ in enumerate(data):
        big_X_train[i], big_X_test[i], big_y_train[i], big_y_test[i] = prepare_data_standard_for_user_split(u_, size)

    return big_X_train, big_X_test, big_y_train, big_y_test


def prepare_data_standard_for_user_split(data, size):
    X_train = [None] * len(data[0])
    X_test = [None] * len(data[0])
    y_train = [None] * len(data[0])
    y_test = [None] * len(data[0])

    for j in range(len(data[0])):
        X_train[j], X_test[j], y_train[j], y_test[j] = train_test_split(data[0][j], data[1][j],
                                                                        test_size=size, random_state=42)

    return X_train, X_test, y_train, y_test


def prepare_data_standard_for_user(data):
    X_train = [None] * len(data[0])
    y_train = [None] * len(data[0])

    for j in range(len(data[0])):
        X_train[j], y_train[j] = data[0][j], data[1][j]

    return X_train, y_train


def prepare_data_standard_from_list(data):
    big_X = [None] * len(data)
    big_y = [None] * len(data)
    for i, u_ in enumerate(data):
        big_X[i], big_y[i] = prepare_data_standard_for_user(u_)
    return big_X, big_y


###################################################################
###################### Filtering functions ########################
###################################################################

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # noinspection PyTupleAssignmentBalance
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs=250, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def remove_eog(data):
    EEG_only = data[:, :-3]
    return EEG_only
