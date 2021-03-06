# TODO: Cropped training
import keras
from models import EEGNet_org
from helper3 import prediction_vector, get_metrics_and_plots
from helper3 import metrics_to_csv
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import time
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from models import EEGNet_var

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
        model = EEGNet_org(nb_classes=len(class_names), Chans=ch_num, Samples=1125, dropoutRate=0.2)

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])

        filename_ = '{0}{1}{2}'.format(model_name, 'standard_user_{}'.format(i + 1), addon)
        metrics[i] = standard_unit(model, features_train[i], labels_train[i],
                                   features_test[i], labels_test[i], filename_,
                                   class_names)

    metrics_to_csv(metrics, '{}_Standard_testing_{}'.format(model_name, addon))


def opt_Dropout_rate_CV_EEGNet(dropout_start, dropout_stop, model_name, big_X_train, big_y_train,
                               big_X_val, big_y_val, class_names=class_names, ch_num=25):
    for i in np.arange(dropout_start, dropout_stop, 0.01):
        EEGnet = EEGNet_org(nb_classes=2, Chans=ch_num, Samples=1125, dropoutRate=i)

        standard_all(EEGnet, model_name, big_X_train, big_y_train, big_X_val, big_y_val,
                     ch_num=ch_num, class_names=class_names,
                     addon='Dropout_{0:.2f}'.format(i))


def full_distributed(model_name, big_X_train, big_y_train, big_X_test, big_y_test,
                     class_names=class_names, ch_num=25, dr=0.1, addon=''):

    features_train = [None] * len(big_X_train)
    labels_train = [None] * len(big_y_train)

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
        model = EEGNet_org(nb_classes=len(class_names), Chans=ch_num, Samples=1125, dropoutRate=0.2)

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])

        filename_ = '{0}{1}{2}'.format(model_name, 'Distributed_{}'.format(i + 1), addon)
        metrics[i] = distributed_unit(model, full_features, full_labels,
                                      features_test[i], labels_test[i],
                                      filename_, class_names)

    metrics_to_csv(metrics, '{}_Distributed_Learning_{}'.format(model_name, filename_))


def incremental_distributed_learning(model_name, big_X_train, big_y_train, big_X_test,
                                     big_y_test, class_names=class_names, ch_num=25):
    users = [1, 2, 3, 4, 5, 6, 7, 8]

    for i in users:
        full_distributed(model_name, big_X_train[0:i], big_y_train[0:i], big_X_test,
                         big_y_test, class_names, ch_num, addon='_{}_users_training'.format(i))


def freezing_layers(model_name, big_X_train, big_y_train, big_X_test, big_y_test,
                    class_names=class_names, ch_num=25, dr=0.1, addon='', fz_layers=-2):
    ###
    ### First create the sequence of training users and single test user
    ###
    list_element = []
    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    series = []

    for idx in range(90):
        if idx % 10 != 0:
            list_element.append(my_list[idx % len(my_list)])
        elif idx % 10 == 0:
            series.append(list_element)
            list_element = []
    series[0] = list_element

    metrics = np.zeros(((len(my_list)), 4))
    ###
    ### Iterate trough the sequences
    ###
    for user_list in series:
        print('Starting Frozen learning for user {}'.format(user_list[-1] + 1))

        model = EEGNet_org(nb_classes=2, Chans=ch_num, Samples=1125, dropoutRate=dr)

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])

        features_train = []
        labels_train = []

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

        # Flatten the data for training
        full_features = np.vstack(features_train)
        full_labels = np.vstack(labels_train)

        ###
        ### Create the second training data (of the specific user)
        ###
        temp = [item for sublist in big_X_train[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_train[user_list[-1]] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_train_2 = np_utils.to_categorical(encoded_Y)

        ###
        ### Create the testing data (of the specific user)
        ###
        temp = [item for sublist in big_X_test[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_test[user_list[-1]] for item in sublist]
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
        del features_train, labels_train, features_train_2, \
            features_test_2, labels_train_2, labels_test_2, \
            full_features, full_labels, model

    metrics_to_csv(metrics, '{}_Frozen_Learning_{}'.format(model_name, filename_))


def freezing_unit(model, full_features, full_labels,
                  features_train_2, features_test_2,
                  labels_train_2, labels_test_2,
                  filename, class_names, fz_layers):
    ###
    ### First train the data on all the users -1
    ###
    start = time.time()

    model.fit(full_features, full_labels,
              batch_size=32, epochs=500, verbose=0)

    end = time.time()
    print('First model training time: {}'.format(end - start))
    model.save('models/{}.h5'.format(filename))

    # # Deletes the existing model
    # del model
    #
    # # Returns a compiled model identical to the previous one
    # model = load_model('my_model.h5')

    ###
    ### Freeze all layers, except for the last two ones
    ###
    for layer in model.layers[:fz_layers]:
        layer.trainable = False

    start = time.time()

    history = model.fit(features_train_2, labels_train_2,
                        batch_size=32, epochs=50, verbose=0,
                        validation_data=(features_test_2, labels_test_2))

    end = time.time()

    print('Second model training time: {}'.format(end - start))

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
    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    series = []

    for idx in range(90):
        if idx % 10 != 0:
            list_element.append(my_list[idx % len(my_list)])
        elif idx % 10 == 0:
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

        model = EEGNet_org(nb_classes=2, Chans=ch_num, Samples=1125, dropoutRate=dr)

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
        temp = [item for sublist in big_X_train[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_train[user_list[-1]] for item in sublist]
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
        temp = [item for sublist in big_X_test[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_test[user_list[-1]] for item in sublist]
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


def frozen_with_model(model_name, big_X_train, big_y_train, big_X_test, big_y_test,
                      class_names=class_names, ch_num=25, ep=0.1, addon='', fz_layers=-2):
    ###
    ### First create the sequence of training users and single test user
    ###
    list_element = []
    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    series = []

    for idx in range(90):
        if idx % 10 != 0:
            list_element.append(my_list[idx % len(my_list)])
        elif idx % 10 == 0:
            series.append(list_element)
            list_element = []
    series[0] = list_element

    metrics = np.zeros(((len(my_list)), 4))
    ###
    ### Iterate trough the sequences
    ###
    r_series = reversed(series)
    for user_list in r_series:
        print('Starting Frozen learning for user {}'.format(user_list[-1] + 1))

        ###
        ### Only create the second training data (of the specific user)
        ###
        temp = [item for sublist in big_X_train[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_train[user_list[-1]] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_train_2 = np_utils.to_categorical(encoded_Y)

        ###
        ### Create the testing data (of the specific user)
        ###
        temp = [item for sublist in big_X_test[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_test[user_list[-1]] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_test_2 = np_utils.to_categorical(encoded_Y)

        filename_ = '{0}{1}{2}{3}{4}'.format(model_name, 'Freezing_{}'.format(user_list[-1] + 1), addon,
                                             '_{}_Frozen'.format(str(fz_layers)), '_{}_Epochs'.format(str(ep)))

        model_file = 'models/{0}{1}_50_Epochs{2}.h5'.format(model_name, 'Freezing_{}'.format(user_list[-1] + 1),
                                                     '_-2_Frozen')

        metrics[user_list[-1]] = freezing_unit_with_model(model_file, features_train_2, features_test_2,
                                                          labels_train_2, labels_test_2,
                                                          filename_, class_names, fz_layers, ep)

        del features_train_2, features_test_2, \
            labels_train_2, labels_test_2

    metrics_to_csv(metrics, '{0}_Frozen_Learning_{1}_FzLayers_{2}_Epochs'.format(model_name, fz_layers, ep))


def freezing_unit_with_model(model_file, features_train_2, features_test_2,
                             labels_train_2, labels_test_2,
                             filename, class_names, fz_layers,
                             ep=50):
    # Returns the compiled model
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(model_file)

    ###
    ### Freeze all layers, except for the last two ones
    ###
    for layer in model.layers[:fz_layers]:
        layer.trainable = False

    start = time.time()

    history = model.fit(features_train_2, labels_train_2,
                        batch_size=32, epochs=ep, verbose=0,
                        validation_data=(features_test_2, labels_test_2))

    end = time.time()

    print('Model training time: {}'.format(end - start))

    y_prob = model.predict(features_test_2)
    values = get_metrics_and_plots(labels_test_2, y_prob, history,
                                   filename, class_names)

    return values


def splited_with_model(model_name, big_X_train, big_y_train, big_X_test, big_y_test,
                        class_names=class_names, ch_num=25, ep=50, addon=''):
    ###
    ### First create the sequence of training users and single test user
    ###
    list_element = []
    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    series = []

    for idx in range(90):
        if idx % 10 != 0:
            list_element.append(my_list[idx % len(my_list)])
        elif idx % 10 == 0:
            series.append(list_element)
            list_element = []
    series[0] = list_element

    metrics = np.zeros(((len(my_list)), 4))
    ###
    ### Iterate trough the sequences
    ###
    r_series = reversed(series)
    for user_list in r_series:
        print('Starting Splited learning for user {}'.format(user_list[-1]))

        ###
        ### Only create the second training data (of the specific user)
        ###
        temp = [item for sublist in big_X_train[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_train[user_list[-1]] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_train_2 = np_utils.to_categorical(encoded_Y)

        ###
        ### Create the testing data (of the specific user)
        ###
        temp = [item for sublist in big_X_test[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test_2 = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_test[user_list[-1]] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_test_2 = np_utils.to_categorical(encoded_Y)

        filename_ = '{0}{1}{2}{3}'.format(model_name, '_Split_{}'.format(user_list[-1] + 1), addon,
                                          '_{}_Epochs'.format(str(ep)))

        model_file = 'models/{0}{1}_50_Epochs{2}.h5'.format(model_name, 'Freezing_{}'.format(user_list[-1] + 1),
                                                            '_-2_Frozen')

        metrics[user_list[-1]] = split_unit_with_model(model_file, features_train_2, features_test_2,
                                                          labels_train_2, labels_test_2,
                                                          filename_, class_names, ep)

        del features_train_2, features_test_2, \
            labels_train_2, labels_test_2

    metrics_to_csv(metrics, '{}_Split_Learning_{}'.format(model_name, filename_))


def split_unit_with_model(model_file, features_train_2, features_test_2,
                          labels_train_2, labels_test_2,
                          filename, class_names, ep=50):
    # Returns the compiled model
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(model_file)

    start = time.time()

    history = model.fit(features_train_2, labels_train_2,
                        batch_size=32, epochs=ep, verbose=0,
                        validation_data=(features_test_2, labels_test_2))

    end = time.time()

    print('Model training time: {}'.format(end - start))

    y_prob = model.predict(features_test_2)
    values = get_metrics_and_plots(labels_test_2, y_prob, history,
                                   filename, class_names)

    return values

def full_freezing(model_name, big_X_train, big_y_train, big_X_test, big_y_test,
                    class_names=class_names, ch_num=25, dr=0.1, addon=''):
    ###
    ### Take all the data for training
    ###

    ###
    ### Train and save the model
    ###

    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    model = EEGNet_org(nb_classes=2, Chans=ch_num, Samples=1125, dropoutRate=dr)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])

    features_train = []
    labels_train = []

    for i in my_list:
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

        # Flatten the data for training
        full_features = np.vstack(features_train)
        full_labels = np.vstack(labels_train)

    filename_ = '{0}{1}{2}'.format(model_name, '_Full_Freezing', addon)

    full_freezing_unit(model, full_features, full_labels,
                       filename_, class_names)


def full_freezing_unit(model, full_features, full_labels,
                  filename, class_names):
    ###
    ### First train the data on all the users -1
    ###
    start = time.time()

    model.fit(full_features, full_labels,
              batch_size=32, epochs=500, verbose=0)

    end = time.time()
    print('First model training time: {}'.format(end - start))
    model.save('models/{}.h5'.format(filename))

    model.save_weights('models/{}_w.h5'.format(filename))

    # # Deletes the existing model
    # del model
    #
    # # Returns a compiled model identical to the previous one
    # model = load_model('my_model.h5')


def transfer_unit(model, training_data, training_labels,
                  testing_data, testing_labels, filename,
                  class_names=class_names):

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])

    # for layer in model.layers[:-7]:
    #     layer.trainable = False

    history = model.fit(training_data, training_labels,
                        batch_size=32, epochs=200, verbose=0,
                        validation_data=(testing_data, testing_labels))

    y_prob = model.predict(testing_data)
    values = get_metrics_and_plots(testing_labels, y_prob, history,
                                   filename, class_names)

    return values

def frozen_4mvt_2mvt(model_name, model_file, big_X_train, big_y_train, big_X_test, big_y_test,
                     class_names=class_names, ch_num=25, ep=50, dr=0.1, addon=''):
    ###
    ### First create the sequence of training users and single test user
    ###
    list_element = []
    my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    series = []

    for idx in range(90):
        if idx % 10 != 0:
            list_element.append(my_list[idx % len(my_list)])
        elif idx % 10 == 0:
            series.append(list_element)
            list_element = []
    series[0] = list_element

    metrics = np.zeros(((len(my_list)), 4))
    ###
    ### Iterate trough the sequences
    ###
    for user_list in series:
        print('Starting Frozen learning for user {}'.format(user_list[-1] + 1))

        features_train = []
        labels_train = []

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
        temp = [item for sublist in big_X_train[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_train.append(temp.reshape(temp.shape[0], 1, ch_num, 1125))
        lab = [item for sublist in big_y_train[user_list[-1]] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_train.append(np_utils.to_categorical(encoded_Y))

        # Flatten the data for training
        full_features = np.vstack(features_train)
        full_labels = np.vstack(labels_train)

        ###
        ### Create the testing data (of the specific user)
        ###
        temp = [item for sublist in big_X_test[user_list[-1]] for item in sublist]
        temp = np.asanyarray(temp)
        temp = np.swapaxes(temp, 1, 2)
        features_test = temp.reshape(temp.shape[0], 1, ch_num, 1125)
        lab = [item for sublist in big_y_test[user_list[-1]] for item in sublist]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(lab)
        encoded_Y = encoder.transform(lab)
        # convert integers to dummy variables (i.e. one hot encoded)
        labels_test = np_utils.to_categorical(encoded_Y)

        filename_ = '{0}{1}{2}'.format(model_name, '_frozen_transfer_learning_{}_'.format(i + 1), addon)

        model_ = EEGNet_var(model_file, 4, 2, Chans=ch_num, dropoutRate=dr, Samples=1125)

        metrics[user_list[-1]] = transfer_unit(model_, full_features, full_labels,
                                               features_test, labels_test, filename_,
                                               class_names)

    metrics_to_csv(metrics, '{}_transfer_frozen_learning_{}'.format(model_name, addon))



def standard_4mvt_2mvt(model_name, model_file, big_X_train, big_y_train, big_X_test, big_y_test,
                     class_names=class_names, ch_num=25, ep=50, dr=0.1, addon=''):
    features_train = [None] * len(big_X_train)
    labels_train = [None] * len(big_y_train)

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
        model_ = EEGNet_var(model_file, 4, 2, Chans=ch_num, dropoutRate=dr, Samples=1125)

        model_.compile(loss=categorical_crossentropy,
                       optimizer=Adam(), metrics=['accuracy'])

        filename_ = '{0}{1}{2}'.format(model_name, '_transfer_standard_learning_{}_'.format(i + 1), addon)
        metrics[i] = transfer_unit(model_, features_train[i], labels_train[i],
                                   features_test[i], labels_test[i], filename_,
                                   class_names)

    metrics_to_csv(metrics, '{}_transfer_standard_learning_{}'.format(model_name, addon))