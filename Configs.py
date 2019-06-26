from easydict import EasyDict as edict
import os, time

param = edict()

# ===== GLOBAL CONFIGURATION =====
'''
1. model_name: 'CNNLSTM' or 'CNNDTRN'.
2. gpu_id:      A number character.
3. norm_height: Normalized image height. The height of input images has already been normalized to 128.
4. image_dir:   The directory path where the input images exist.
5. alphabet:    The file that includes the character set.
6. num_classes: The length of the character set + 1 (blank)
7. saving_path: The directory that saves trained models and the log file.
'''

param['model_name'] = 'CNNLSTM' # 'CNNLSTM' or 'CNNDTRN'
param['gpu_id'] = '0'
param['norm_height'] = 128
param['image_dir'] = '<path_to_IAM_data>/data/imgs/lines_h128/'
param['alphabet'] = 'DataLoader/alphabet_iam.txt'

f = open(param.alphabet, 'r')
l = f.readlines()[0]
f.close()
param['num_classes'] = len(l) + 1

param['saving_path'] = os.path.join('models',
                                    param.model_name,
                                    time.strftime('%Y%m%d-%H:%M:%S', time.localtime()))


#===== TRAINING CONFIGURATION =====
'''
1. random_seed:        The random seed for random, numpy.random and pytorch.random, etc.
2. train_list:         The file contains paths of training images and their ground truths.
3. val_list:           The file contains paths of validation images and their ground truths.
4. batch_size_tr:      Training batch size.
5. batch_size_va:      Validation batch size.
6. continue_train:     Whether to load a saved model.
7. continue_path:      The path to the continue model.
8. num_epochs:         The number of training epochs.
9. display_decoded:    Whether to display the prediction results during training.
10. display_interval:  The step interval to display the prediction results during training.
11. learning_rate:     The initial learning rate.
12. clip:              Whether to clip the large gradients.
13. clip_norm:         The max norm of gradients.
14. l2_weight_decay:   The weight decay rate.
15. lr_reduce_patient: If validation CER doesn't decrease for consecutive lr_reduce_patient epochs, 
                       then reducing the learning rate.
16. lr_factor:         lr <== lr_factor * lr if we reduce the learning rate. 
'''

param['train'] = edict()
param.train['random_seed'] = 2
param.train['train_list'] = '<path_to_IAM_data>/data/lang/lines/char/aachen/tr.txt'
param.train['val_list'] = '<path_to_IAM_data>/data/lang/lines/char/aachen/va.txt'
param.train['batch_size_tr'] = 16
param.train['batch_size_va'] = 16
param.train['continue_train'] = False
param.train['continue_path'] = ''
param.train['num_epochs'] = 350
param.train['display_decoded'] = False
param.train['disp_interval'] = 10
param.train['learning_rate'] = 3e-4
param.train['clip'] = True
param.train['clip_norm'] = 50
param.train['l2_weight_decay'] = 0
param.train['lr_reduce_patient'] = 15
param.train['lr_factor'] = 0.6


#===== NETWORK CONFIGURATION =====
param['net'] = edict()

#---- CNN ----
'''
1. conv_knum:      The number of kernel in each CNN layer.
2. conv_dropout:   The dropout rate of each CNN layer.
3. con_ksize:      The kernel size of each CNN layer.
4. conv_stride:    The stride of each CNN layer.
5. conv_padding:   The padding size of each CNN layer.
6. conv_with_pool: Whether to do MaxPooling after each CNN layer.
7. in_dim:         The dimension of input that equals to param.norm_height.
'''

param.net['cnn'] = edict()
param.net.cnn['conv_knum'] = [16, 32, 48, 64, 80]
param.net.cnn['conv_dropout'] = [0, 0, 0.2, 0.2, 0.2]
param.net.cnn['conv_ksize'] = [3, 3, 3, 3, 3]
param.net.cnn['conv_stride'] = [1, 1, 1, 1, 1]
param.net.cnn['conv_padding'] = [1, 1, 1, 1, 1]
param.net.cnn['conv_with_pool'] = [True, True, True, False, False]
param.net.cnn['in_dim'] = param.norm_height

#---- LSTM ----
'''
1. hidden_size:    The number of units in each LSTM layer.
2. num_layers:     The number of LSTM layer.
3. dropout:        The dropout rate of each LSTM layer.
4. bidirectional:  Whether to use bidirectional LSTM.
5. out_dimension:  The dimension of output (after fc layer). Equals to param.num_classes
6. linear_dropout: The dropout rate of the fc layer.
'''

param.net['rnn'] = edict()
param.net.rnn['hidden_size'] = 256
param.net.rnn['num_layers'] = 5
param.net.rnn['dropout'] = 0.5
param.net.rnn['bidirectional'] = True
param.net.rnn['out_dim'] = param.num_classes
param.net.rnn['linear_dropout'] = 0.5

#---- DTRN ----
'''
DCG: Dynamic Coefficients Generator (The secondary network)
TRN: Temporal Residual Network (The primary network)

1. dcg_hidden_size:     The number of units in DCG.
2. dcg_dropoutout:      The dropout rate of DCG.
3. trn_hidden_size:     The number of units in each TRN layer.
4. trn_discount_factor: The discount factor (gamma).
5. trn_dropout:         The dropout rate of each TRN layer.
6. bidirectional:       Whether to use bidirectional DTRN.
7. outdimension:        The dimension of output (after fc layer). Equals to param.num_classes
8. linear_dropout:      The dropout rate of the fc layer.
'''

param.net['dtrn'] = edict()
param.net.dtrn['dcg_hidden_size'] = 128
param.net.dtrn['dcg_dropout'] = 0.5
param.net.dtrn['trn_hidden_size'] = [256] * 5
param.net.dtrn['trn_discount_factor'] = 0.4
param.net.dtrn['trn_dropout'] = [0.5] * 5
param.net.dtrn['out_dim'] = param.num_classes
param.net.dtrn['linear_dropout'] = 0.5


#===== TEST CONFIGURATION =====
'''
1. test_list:     The file contains paths of test images and their ground truths.
2. batch_size_te: Test batch size.
3. model_path:    Path to model to be tested.
'''

param['test'] = edict()
param.test['test_list'] = '<path_to_IAM_data>/data/lang/lines/char/aachen/te.txt'
param.test['batch_size_te'] = 16
param.test['model_path'] = 'models/CNNLSTM/yearmonthday-00:00:00/m-epoch1.pth.tar'
