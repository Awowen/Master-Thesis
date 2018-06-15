from helper3 import prepare_data_standard_from_list, load_simplify_data
from controllers import freezing_layers, splitted_layers, frozen_with_model, splited_with_model, standard_all
from models import EEGNet_org
import glob as glob

train_list = glob.glob('../Data/BCI_4_2b/*T.mat')
test_list = glob.glob('../Data/BCI_4_2b/*E.mat')

train_list.sort()
test_list.sort()

big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False))

#
# EEGnet = EEGNet_org(nb_classes=2, Chans=13, Samples=1125, dropoutRate=0.1)
#
# standard_all(EEGnet, 'EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#              class_names=['Right hand', 'Feet'], ch_num=13, addon='_13_channels')
# freezing_layers('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                 class_names=['Right hand', 'Feet'],
#                 ch_num=13, dr=0.1, addon='_13_channels')

# for i in range(-1, -20, -1):
#     for j in range(50, 100, 50):
#         print('Freezing {0} \n Epochs {1}'.format(i,j))
#         frozen_with_model('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                           class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                           ch_num=25, ep=j, addon='', fz_layers=i)

# for j in range(50, 100, 50):
#     print('Epochs {}'.format(j))
#     splited_with_model('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                        class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                        ch_num=25, ep=j, addon='')
# frozen_with_model('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                   class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                   ch_num=25, dr=0.1, addon='50_Epochs')

