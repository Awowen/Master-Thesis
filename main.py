import matplotlib

matplotlib.use('Agg')
from helper import prepare_data_standard_from_list, load_simplify_data
from controllers import standard_all, opt_Dropout_rate_CV_EEGNet
import glob as glob

train_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*E.mat')
test_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*T.mat')

# train_list = glob.glob('Data/A*E.mat')
# test_list = glob.glob('Data/A*T.mat')

train_list.sort()
test_list.sort()



# Without EOG, with filtering
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, True))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, True))
#
opt_Dropout_rate_CV_EEGNet(.2, .3, 'EEGNet_filtered_wo_EOG_22_ch', big_X_train, big_y_train, big_X_test, big_y_test,
                           class_names=['Left hand', 'Right hand',
                                        'Both Feet', 'Tongue'], ch_num=22)
# With EOG, with filtering
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False))
opt_Dropout_rate_CV_EEGNet(.2, .3, 'EEGNet_filtered_w_EOG_25_ch', big_X_train, big_y_train, big_X_test, big_y_test,
                           class_names=['Left hand', 'Right hand',
                                        'Both Feet', 'Tongue'], ch_num=25)
########################################################################################################################
# Only look at the channels Fz, C3, Cz, C4, Fz
tested_channels = [0, 7, 9, 11, 19]
# Without EOG, with filtering, selected channels
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, True, tested_channels))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, True, tested_channels))

opt_Dropout_rate_CV_EEGNet(.2, .3, 'EEGNet_filtered_wo_EOG_5_ch', big_X_train, big_y_train, big_X_test, big_y_test,
                           class_names=['Left hand', 'Right hand',
                                        'Both Feet', 'Tongue'], ch_num=len(tested_channels))
# With EOG, with filtering, selected channels
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False, tested_channels))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False, tested_channels))

opt_Dropout_rate_CV_EEGNet(.2, .3, 'EEGNet_filtered_w_EOG_8_ch', big_X_train, big_y_train, big_X_test, big_y_test,
                           class_names=['Left hand', 'Right hand',
                                        'Both Feet', 'Tongue'], ch_num=len(tested_channels)+3)
