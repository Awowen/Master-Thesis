import matplotlib

matplotlib.use('Agg')
from helper import prepare_data_standard_from_list, load_simplify_data
from controllers import freezing_layers, splitted_layers, frozen_with_model, splited_with_model, standard_all, full_distributed, full_freezing, frozen_from_2mvt
from controllers import standard_from_2mvt
from models import EEGNet_org
import glob as glob

train_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*E.mat')
test_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*T.mat')

#train_list = glob.glob('../Data/A*E.mat')
#test_list = glob.glob('../Data/A*T.mat')

train_list.sort()
test_list.sort()

tested_channels = [7, 9, 11,
                   22, 23, 24]


# Freezing Learning
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False, tested_channels))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False, tested_channels))

# EEGnet = EEGNet_org(nb_classes=4, Chans=6, Samples=1125, dropoutRate=0.1)

# standard_all(EEGnet, 'EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#              class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#              ch_num=6, addon='_6_channels_4_classes')
#
# full_distributed('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                  class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                  ch_num=6, dr=0.1, addon='_6_Channels_4_classes')
#
#
# freezing_layers('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                 class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                 ch_num=6, dr=0.1, addon='_6_Channels_4_classes')
#
# full_freezing('EEG_Net', big_X_train, big_y_train, big_X_test, big_y_test,
#               class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#               ch_num=6, dr=0.1, addon='_6_Channels_4_classes')

frozen_from_2mvt('EEG_net', 'models/EEG_Net_Full_Freezing_6_Channels_2_classes.h5', big_X_train, big_y_train, big_X_test, big_y_test,
                 class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'], ch_num=6,
                 ep=200, dr=0.1, addon='_6_channels_7L_Frozen')

standard_from_2mvt('EEG_net', 'models/EEG_Net_Full_Freezing_6_Channels_2_classes.h5', big_X_train, big_y_train, big_X_test, big_y_test,
                 class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'], ch_num=6,
                 ep=200, dr=0.1, addon='_6_channels_7L_Frozen')


# frozen_with_model('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                   class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                   ch_num=25, dr=0.1, addon='50_Epochs')
