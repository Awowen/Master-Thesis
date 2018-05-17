# TODO: Import models, helper
# TODO: distributed learning
# TODO: full distributed learning
# TODO: full test
# TODO: partial
# TODO: standard ML
# TODO: Cropped training
from models import EEGNet_org
from helper import prediction_vector, get_metrics_and_plots
from helper import metrics_to_csv
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# load_simplify_data(filenames) --> to put in main
class_names = ['Left hand', 'Right hand',
               'Both Feet', 'Tongue']


def standard_unit(model, training_data, training_labels,
                  testing_data, testing_labels, filename,
                  class_names=class_names):

    history = model.fit(training_data, training_labels,
                       batch_size=32, epochs=500, verbose=0,
                       validation_data=(testing_data, testing_labels))
    y_prob = model.predict(testing_data)
    values = get_metrics_and_plots(testing_labels, y_prob, history,
                                   filename, class_names)

    return values


def distributed_unit(model, training_data, training_labels,
                     testing_data, testing_labels, filename,
                     class_names=class_names):

    history = model.fit(training_data, training_labels,
                        batch_size=32, epochs=500, verbose=0,
                        validation_data=(testing_data, testing_labels))

    y_prob = model.predict(testing_data)
    values = get_metrics_and_plots(testing_labels, y_prob, history,
                                   filename, class_names)

    return values


def standard_all(model_, model_name, big_X_train, big_y_train,
                 big_X_test, big_y_test, ch_num, class_names=class_names, addon=''):

    features_train = [None] * len(big_X_train)
    labels_train = [None] * len(big_y_train)

    model_.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])

    for i, (X_user, y_user) in enumerate(zip(big_X_train, big_y_train)):
        temp = [item for sublist in X_user for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train[i] = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in y_user for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_train[i] = np_utils.to_categorical(encoded_Y)

    features_test = [None] * len(big_X_test)
    labels_test = [None] * len(big_y_test)
    for i, (X_user, y_user) in enumerate(zip(big_X_test, big_y_test)):
        temp = [item for sublist in X_user for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test[i] = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in y_user for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_test[i] = np_utils.to_categorical(encoded_Y)

    metrics = np.zeros(((len(big_y_train)), 4))

    for i in range(len(big_X_train)):
        filename_ = '{0}{1}{2}'.format(model_name, 'standard_user_{}'.format(i + 1), addon)
        metrics[i] = standard_unit(model_, features_train[i], labels_train[i],
                                   features_test[i], labels_test[i], filename_,
                                   class_names)

    metrics_to_csv(metrics, '{}_Standard_testing_{}'.format(model_name, addon))


def opt_Dropout_rate_CV_EEGNet(dropout_start, dropout_stop, model_name, big_X_train, big_y_train,
                               big_X_val, big_y_val, class_names=class_names, ch_num=25):

    for i in np.arange(dropout_start, dropout_stop, 0.01):

        EEGnet = EEGNet_org(nb_classes=4, Chans=ch_num, Samples=1125, dropoutRate=i)

        standard_all(EEGnet, model_name, big_X_train, big_y_train, big_X_val, big_y_val,
                     ch_num=ch_num, class_names=class_names,
                     addon='Dropout_{0:.2f}'.format(i))

def full_distributed(model_name, big_X_train, big_y_train, big_X_test, big_y_test,
                     class_names=class_names, ch_num=25, dr=0.1, addon=''):
    model = EEGNet_org(nb_classes=4, Chans=ch_num, Samples=1125, dropoutRate=dr)

    features_train = [None] * len(big_X_train)
    labels_train = [None] * len(big_y_train)

    model.compile(loss=categorical_crossentropy,
                   optimizer=Adam(), metrics=['accuracy'])

    for i, (X_user, y_user) in enumerate(zip(big_X_train, big_y_train)):
        temp = [item for sublist in X_user for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train[i] = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in y_user for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_train[i] = np_utils.to_categorical(encoded_Y)

    features_test = [None] * len(big_X_test)
    labels_test = [None] * len(big_y_test)
    for i, (X_user, y_user) in enumerate(zip(big_X_test, big_y_test)):
        temp = [item for sublist in X_user for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test[i] = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in y_user for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_test[i] = np_utils.to_categorical(encoded_Y)

    # Flatten the data for training
    full_features = np.vstack(features_train)
    full_labels = np.vstack(labels_train)

    metrics = np.zeros(((len(big_y_train)), 4))

    for i in range(len(big_X_train)):
        filename_ = '{0}{1}{2}'.format(model_name, 'Distributed_{}'.format(i + 1), addon)
        metrics[i] = distributed_unit(model, full_features, full_labels,
                                      features_test[i], labels_test[i],
                                      filename_, class_names)

    metrics_to_csv(metrics, '{}_Distributed_Learning_{}'.format(model_name, filename_))

def incremental_distributed_learning(model_name, big_X_train, big_y_train, big_X_test,
                                     big_y_test, class_names=class_names, ch_num=25):

    users = [1,2,3,4,5,6,7,8]

    for i in users:
        full_distributed(model_name, big_X_train[0:i], big_y_train[0:i], big_X_test,
                         big_y_test, class_names, ch_num, addon='_{}_users_training'.format(i))


def freezing_layers(model_name, big_X_train, big_y_train, big_X_test, big_y_test,
                    class_names=class_names, ch_num=25, dr=0.1, addon='', fz_layers=-2):

    ###
    ### First create the sequence of training users and single test user
    ###
    list_element = []
    my_list = [0, 1, 2, 3, 4, 5, 6, 7]
    series = []

    for idx in range(72):
        if idx % 9 != 0:
            list_element.append(my_list[idx % len(my_list)])
        elif idx % 9 == 0:
            series.append(list_element)
            list_element = []
    series[0] = list_element

    features_train = []
    labels_train = []

    metrics = np.zeros(((len(my_list)), 4))
    ###
    ### Iterate trough the sequences
    ###
    for user_list in series:
        print('Starting Frozen learning for user {}'.format(user_list[-1]))

        model = EEGNet_org(nb_classes=4, Chans=ch_num, Samples=1125, dropoutRate=dr)

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])

        # Create the first training data
        for i in user_list[:-1]:

            temp = [item for sublist in big_X_train[i] for item in sublist]
            temp = np.asanyarray(temp)
            temp = np.swapaxes(temp, 1, 2)
            features_train.append(temp.reshape(temp.shape[0], 1, ch_num, 1125))
            lab = [item for sublist in big_y_train[i] for item in sublist]
            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(lab)
            encoded_Y = encoder.transform(lab)
            # convert integers to dummy variables (i.e. one hot encoded)
            labels_train.append(np_utils.to_categorical(encoded_Y))


            # Also add the testing data to increase the size of training data
            temp = [item for sublist in big_X_test[i] for item in sublist]
            temp = np.asanyarray(temp)
            temp = np.swapaxes(temp, 1, 2)
            features_train.append(temp.reshape(temp.shape[0], 1, ch_num, 1125))
            lab = [item for sublist in big_y_test[i] for item in sublist]
            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(lab)
            encoded_Y = encoder.transform(lab)
            # convert integers to dummy variables (i.e. one hot encoded)
            labels_train.append(np_utils.to_categorical(encoded_Y))
        ###
        ### Create the second training data (of the specific user)
        ###
        temp = [item for sublist in big_X_train[-1] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_train[-1] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_train_2 = np_utils.to_categorical(encoded_Y)

        # Flatten the data for training
        full_features = np.vstack(features_train)
        full_labels = np.vstack(labels_train)
        ###
        ### Create the testing data (of the specific user)
        ###
        temp = [item for sublist in big_X_test[-1] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_test[-1] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_test_2 = np_utils.to_categorical(encoded_Y)

        filename_ = '{0}{1}{2}{3}'.format(model_name, 'Freezing_{}'.format(user_list[-1] + 1), addon,
                                          '_{}_Frozen'.format(str(fz_layers)))

        metrics[user_list[-1]] = freezing_unit(model, full_features, full_labels,
                                               features_train_2, features_test_2,
                                               labels_train_2, labels_test_2,
                                               filename_, class_names, fz_layers)
    metrics_to_csv(metrics, '{}_Frozen_Learning_{}'.format(model_name, filename_))


def freezing_unit(model, full_features, full_labels,
                  features_train_2, features_test_2,
                  labels_train_2, labels_test_2,
                  filename, class_names, fz_layers):
    ###
    ### First train the data on all the users -1
    ###
    model.fit(full_features, full_labels,
              batch_size=32, epochs=500, verbose=0)

    ###
    ### Freeze all layers, except for the last two ones
    ###
    for layer in model.layers[:fz_layers]:
        layer.trainable = False

    history = model.fit(features_train_2, labels_train_2,
                        batch_size=32, epochs=500, verbose=0,
                        validation_data=(features_test_2, labels_test_2))

    y_prob = model.predict(features_test_2)
    values = get_metrics_and_plots(labels_test_2, y_prob, history,
                                   filename, class_names)

    return values


def splitted_layers(model_name, big_X_train, big_y_train, big_X_test, big_y_test,
                    class_names=class_names, ch_num=25, dr=0.1, addon=''):

    ###
    ### First create the sequence of training users and single test user
    ###
    list_element = []
    my_list = [0, 1, 2, 3, 4, 5, 6, 7]
    series = []

    for idx in range(72):
        if idx % 9 != 0:
            list_element.append(my_list[idx % len(my_list)])
        elif idx % 9 == 0:
            series.append(list_element)
            list_element = []
    series[0] = list_element

    features_train = []
    labels_train = []

    metrics = np.zeros(((len(my_list)), 4))
    ###
    ### Iterate trough the sequences
    ###
    for user_list in series:
        print('Starting Splitted learning for user {}'.format(user_list[-1]))

        model = EEGNet_org(nb_classes=4, Chans=ch_num, Samples=1125, dropoutRate=dr)

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])

        # Create the first training data
        for i in user_list[:-1]:

            temp = [item for sublist in big_X_train[i] for item in sublist]
            temp = np.asanyarray(temp)
            temp = np.swapaxes(temp, 1, 2)
            features_train.append(temp.reshape(temp.shape[0], 1, ch_num, 1125))
            lab = [item for sublist in big_y_train[i] for item in sublist]
            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(lab)
            encoded_Y = encoder.transform(lab)
            # convert integers to dummy variables (i.e. one hot encoded)
            labels_train.append(np_utils.to_categorical(encoded_Y))


            # Also add the testing data to increase the size of training data
            temp = [item for sublist in big_X_test[i] for item in sublist]
            temp = np.asanyarray(temp)
            temp = np.swapaxes(temp, 1, 2)
            features_train.append(temp.reshape(temp.shape[0], 1, ch_num, 1125))
            lab = [item for sublist in big_y_test[i] for item in sublist]
            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(lab)
            encoded_Y = encoder.transform(lab)
            # convert integers to dummy variables (i.e. one hot encoded)
            labels_train.append(np_utils.to_categorical(encoded_Y))
        ###
        ### Create the second training data (of the specific user)
        ###
        temp = [item for sublist in big_X_train[-1] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_train[-1] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_train_2 = np_utils.to_categorical(encoded_Y)

        # Flatten the data for training
        full_features = np.vstack(features_train)
        full_labels = np.vstack(labels_train)
        ###
        ### Create the testing data (of the specific user)
        ###
        temp = [item for sublist in big_X_test[-1] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_test[-1] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_test_2 = np_utils.to_categorical(encoded_Y)

        filename_ = '{0}{1}{2}'.format(model_name, 'Splitted_{}'.format(user_list[-1] + 1), addon)

        metrics[user_list[-1]] = freezing_unit(model, full_features, full_labels,
                                               features_train_2, features_test_2,
                                               labels_train_2, labels_test_2,
                                               filename_, class_names)
    metrics_to_csv(metrics, '{}_Splitted_Learning_{}'.format(model_name, filename_))


def splitted_unit(model, full_features, full_labels,
                  features_train_2, features_test_2,
                  labels_train_2, labels_test_2,
                  filename, class_names):
    ###
    ### First train the data on all the users -1
    ###
    model.fit(full_features, full_labels,
              batch_size=32, epochs=500, verbose=0)

    history = model.fit(features_train_2, labels_train_2,
                        batch_size=32, epochs=500, verbose=0,
                        validation_data=(features_test_2, labels_test_2))

    y_prob = model.predict(features_test_2)
    values = get_metrics_and_plots(labels_test_2, y_prob, history,
                                   filename, class_names)

    return values
