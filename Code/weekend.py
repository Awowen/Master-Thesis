import matplotlib

matplotlib.use('Agg')
from helper import prepare_data_standard_from_list, load_simplify_data
from controllers import freezing_layers, splitted_layers, frozen_with_model, splited_with_model
import glob as glob

train_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*E.mat')
test_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*T.mat')

#train_list = glob.glob('../Data/A*E.mat')
#test_list = glob.glob('../Data/A*T.mat')

train_list.sort()
test_list.sort()

tested_channels = [1, 3, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 15, 17]


# Freezing Learning
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, True, tested_channels))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, True, tested_channels))

# for i in range(5, 40, 5):
#     d = i / 100
#     print('Dropout {}'.format(d))
#     freezing_layers('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                     class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                     ch_num=25, dr=d, addon='_{}_Dropout'.format(str(d).replace('.', '')))

# for i in range(-1, -15, -1):
#     print('Freezing {}'.format(i))
#     freezing_layers('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
#                     class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
#                     ch_num=25, dr=0.1, addon='_01_Dropout', fz_layers=i)

freezing_layers('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
                class_names=['Left hand', 'Right hand', 'Both Feet', 'Tongue'],
                ch_num=13, dr=0.1, addon='_13_channels')


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

