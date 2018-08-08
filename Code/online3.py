import matplotlib

matplotlib.use('Agg')
from helper3 import prepare_data_standard_from_list, load_simplify_data
from controllers3 import freezing_layers, splitted_layers, frozen_with_model, splited_with_model
from controllers3 import standard_all, full_distributed, full_freezing, frozen_4mvt_2mvt, standard_4mvt_2mvt
from models import EEGNet_org
import glob as glob

train_list = glob.glob('../../../opt/shared-data/BCI_Comp_4_2b/B*T.mat')
test_list = glob.glob('../../../opt/shared-data/BCI_Comp_4_2b/B*E.mat')

train_list.sort()
test_list.sort()

# Freezing Learning
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False))


EEGnet = EEGNet_org(nb_classes=2, Chans=6, Samples=1125, dropoutRate=0.1)

# standard_all(EEGnet, 'EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#              class_names=['Left hand', 'Right hand'],
#              ch_num=6, addon='_6_channels')
#
# full_distributed('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                  class_names=['Left hand', 'Right hand'],
#                  ch_num=6, dr=0.1, addon='_6_Channels')

# freezing_layers('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                 class_names=['Left hand', 'Right hand'],
#                 ch_num=6, dr=0.1, addon='_6_Channels')
#
# full_freezing('EEG_Net', big_X_train, big_y_train, big_X_test, big_y_test,
#               class_names=['Left hand', 'Right hand'],
#               ch_num=6, dr=0.1, addon='_6_Channels_2_classes')
#
standard_4mvt_2mvt('EEG_net', 'models/EEG_Net_Full_Freezing_6_Channels_4_classes.h5', big_X_train, big_y_train, big_X_test, big_y_test,
                 class_names=['Left hand', 'Right hand'], ch_num=6,
                 ep=200, dr=0.1, addon='_6_channels')

frozen_4mvt_2mvt('EEG_net', 'models/EEG_Net_Full_Freezing_6_Channels_4_classes.h5', big_X_train, big_y_train, big_X_test, big_y_test,
                 class_names=['Left hand', 'Right hand'], ch_num=6,
                 ep=200, dr=0.1, addon='_6_channels')


# frozen_with_model('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                   class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                   ch_num=25, dr=0.1, addon='50_Epochs')
