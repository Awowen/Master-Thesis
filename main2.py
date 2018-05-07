import matplotlib

matplotlib.use('Agg')
from helper import prepare_data_standard_from_list, load_simplify_data
from controllers import full_distributed, incremental_distributed_learning
import glob as glob

train_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*E.mat')
test_list = glob.glob('../../../opt/shared-data/BCI_Comp_4/A*T.mat')

# train_list = glob.glob('Data/A*E.mat')
# test_list = glob.glob('Data/A*T.mat')

train_list.sort()
test_list.sort()



# Distributed learning
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False))

full_distributed('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
                 class_names=['Left hand', 'Right hand','Both Feet', 'Tongue'],
                 ch_num=25, dr=0.1, addon='_01_Dropout')

# Distributed learning
big_X_train, big_y_train = prepare_data_standard_from_list(load_simplify_data(train_list, True, False))
big_X_test, big_y_test = prepare_data_standard_from_list(load_simplify_data(test_list, True, False))

full_distributed('EEG_net', big_X_train, big_y_train, big_X_test, big_y_test,
                 class_names=['Left hand', 'Right hand','Both Feet', 'Tongue'],
                 ch_num=25, dr=0.2, addon='_02_Dropout')
