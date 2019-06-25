Dynamic Temporal Residual Network for Sequence Modeling
=================================================================================
This is a PyTorch implementation of the DTRN models proposed in ***Dynamic Temporal Residual Network for Sequence Modeling***.

## Usage
### 1. Data Preparation
- Step 1

For IAM and Rimes datasets, we use the tools provided in [Laia System](https://github.com/jpuigcerver/Laia) to preprocess data.
When the preparation is done, a directory named "data" will be generated, with an internal structure similar to the following：

```
<path_to_IAM_data>  
│
└───data
    │
    └───imgs
    |   │
    |   |───lines
    |   └───lines_h128
    |───────lang
    |───────lists
    └───────...
```

- Step 2

To store all the images and labels in a file, run the following command.
```
python preprocess.py --data_root=<path_to_IAM_data>
```
A directory named "precomputed" will be generated in _<path_to_IAM_data>/data/_.


### 2. Training and validation
- Setting configurations
All configurations are in _Configs.py_. Refer to the comments in _Configs.py_ for details. Nevertheless, you can only edit the 20th line:
```
param['data_root'] = <path_to_IAM_data>/data/precomputed
```
and keep other configurations unchanged.

- Train the network
Run the following command to train:
```
python train.py
```
The validation will be conducted after each epoch's training.


### 3. Test
- Setting configurations

Set the model file path in the 151th line in _Configs.py_. For example,
```
param.test['model_path'] = 'models/CNNLSTM/20190631-00:00:00/m-epoch1.pth.tar'
```

- Test

Run the following command to test:
```
python test.py
```

### 4. Training on your own dataset
To train on other datasets, you need to constrcut your own data loader. For each batch, the output of the data loader should contain the following elements.
``` python
inputs, sparse_labels, in_seq_lens, packed_labels, label_sizes = data_for_one_batch
```
- **inputs**

  Input sequence with the shape of [B, C, T, N], where B is bacth size, C is the number of channels, T is the maximum length and N is the feature dimension.

- **sparse_labels**

  Labels in sparse tensor form. Required by the TensorFlow function "tf.edit_distance" for computing CER.

- **in_seq_lens**

  The length of each input of the batch, with the shape of [B].

- **packed_labels**

  Put all labels of the batch in one tensor, with the shape of [total_label_len_of_batch].

- **label_sizes**

  The length of each label of the batch, with the shape of [B]. packed_labels and label_sizes are used to compute CTC loss.

For details, please refer to _DataLoader/IAMDataLoader.py_
