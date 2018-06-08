import matplotlib

matplotlib.use('Agg')
from helper import prepare_data_standard_from_list, load_simplify_data
from controllers import standard_all, opt_Dropout_rate_CV_EEGNet
import glob as glob
from models import EEGNet_org

train_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*E.mat')
test_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*T.mat')

# train_list = glob.glob('Data/A*E.mat')
# test_list = glob.glob('Data/A*T.mat')

train_list.sort()
test_list.sort()



# # With EOG, with filtering
# big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False))
# big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False))
#
# EEGnet = EEGNet_org(nb_classes=4, Chans=25, Samples=1125, dropoutRate=0.1)
#
# standard_all(EEGnet, 'EEGNet_filtered_w_EOG_25_ch_D01', big_X_train, big_y_train, big_X_test, big_y_test,
#                            class_names=['Left hand', 'Right hand',
#                                         'Both Feet', 'Tongue'], ch_num=25)
#
# # Without EOG, with filtering
# big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, True))
# big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, True))
#
# EEGnet = EEGNet_org(nb_classes=4, Chans=22, Samples=1125, dropoutRate=0.1)
#
# standard_all(EEGnet, 'EEGNet_filtered_w_EOG_22_ch_D01', big_X_train, big_y_train, big_X_test, big_y_test,
#                            class_names=['Left hand', 'Right hand',
#                                         'Both Feet', 'Tongue'], ch_num=22)

#
# # With EOG, without filtering
# big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, False, False))
# big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, False, False))
#
# EEGnet = EEGNet_org(nb_classes=4, Chans=25, Samples=1125, dropoutRate=0.2)
#
# standard_all(EEGnet, 'EEGNet_non_filtered_w_EOG_25_ch_D02', big_X_train, big_y_train, big_X_test, big_y_test,
#                            class_names=['Left hand', 'Right hand',
#                                         'Both Feet', 'Tongue'], ch_num=25)

# With EOG, without filtering
# big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False))
# big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False))
# opt_Dropout_rate_CV_EEGNet(.2, .3, 'EEGNet_filtered_w_EOG_25_ch', big_X_train, big_y_train, big_X_test, big_y_test,
#                            class_names=['Left hand', 'Right hand',
#                                         'Both Feet', 'Tongue'], ch_num=25)
# ########################################################################################################################
# Only look at the channels Fz, C3, Cz, C4, Pz
tested_channels = [0, 7, 9, 11, 19]
# # Without EOG, with filtering, selected channels
# big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, True, tested_channels))
# big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, True, tested_channels))
#
# EEGnet = EEGNet_org(nb_classes=4, Chans=len(tested_channels), Samples=1125, dropoutRate=0.1)
#
#
# standard_all(EEGnet, 'EEGNet_filtered_wo_EOG_5_ch_D01', big_X_train, big_y_train, big_X_test, big_y_test,
#                            class_names=['Left hand', 'Right hand',
#                                         'Both Feet', 'Tongue'], ch_num=len(tested_channels))

# With EOG, with filtering, selected channels
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False, tested_channels))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False, tested_channels))

EEGnet = EEGNet_org(nb_classes=4, Chans=8, Samples=1125, dropoutRate=0.1)


standard_all(EEGnet, 'EEGNet_filtered_w_EOG_8_ch_D01', big_X_train, big_y_train, big_X_test, big_y_test,
                           class_names=['Left hand', 'Right hand',
                                        'Both Feet', 'Tongue'], ch_num=8)





