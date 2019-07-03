Dynamic Temporal Residual Network for Sequence Modeling
=================================================================================
This is a PyTorch implementation of the models proposed in ***Dynamic Temporal Residual Network for Sequence Modeling***.

***Note: for now the codes are under sort and there will be possible revision.***

## Requirements
- PyTorch 1.0 or later
- TensorFlow 1.7 or later
- torch_baidu_ctc
- easydict
- pprint
- progressbar

## Usage
We will take the IAM dataset for an example. Other datasets are similar.

### 1. Data Preparation

We use the tools provided in [Laia System](https://github.com/jpuigcerver/Laia) to preprocess data. Please follow its instruction prepare the images and labels.

When the preparation is done, a directory named "data" will be generated, with the structure similar to the following：

```
<path_to_IAM_data>  
│
└───data
    |───imgs
    |   |───lines
    |   └───lines_h128
    |       |───a01-000u-00.jpg
    |       └───...
    |
    |───lang
    |   |───forms
    |   └───lines
    |       └───char
    |           └───aachen
    |               |───tr.txt
    |               |───va.txt
    |               |───te.txt
    |               └───...
    └───...
```

We will use images in _<path_to_IAM_data>/data/imgs/lines_h128_, and ground truths in _<path_to_IAM_data>/data/lang/lines/char/aachen_.


### 2. Training and validation
- Setting configurations

All configurations are in _Configs.py_. Refer to the comments in _Configs.py_ for details. Nevertheless, to run the training program you only need to edit three lines:
```
20th line: param['data_root'] = <path_to_IAM_data>/data/imgs/lines_h128/
56th line: param.train['train_list'] = '<path_to_IAM_data>/data/lang/lines/char/aachen/tr.txt'
57th line: param.train['val_list'] = '<path_to_IAM_data>/data/lang/lines/char/aachen/va.txt'
```
while keeping other configurations unchanged.

- Train the network

Run the following command to train:
```
python train.py
```
The validation will be conducted after each epoch's training.


### 3. Test
- Setting configurations

Set the test file list path and the model file path in _Configs.py_. For example:
```
147th line: param.test['test_list'] = '<path_to_IAM_data>/data/lang/lines/char/aachen/te.txt'
149th line: param.test['model_path'] = 'models/CNNLSTM/20190631-00:00:00/m-epoch1.pth.tar'
```

- Test

Run the following command to test:
```
python test.py
```

### 4. Training on your own dataset
To train on other datasets, you need to constrcut your own data loader. For each batch, the output of the data loader should contain the following elements.
``` python
inputs, sparse_labels, packed_labels, in_seq_lens, label_len = data_for_one_batch
```
- **inputs**

  An array with the shape of [B, C, T, N], where B is bacth size, C is the number of channels, T is the maximum length and N is the feature dimension.

- **label_len**

  An int number. The total length of labels in the batch.

- **sparse_labels**

  A sparse tensor. Required by the TensorFlow function "tf.edit_distance" for computing CER.
  
- **packed_labels**

  A tuple (labels, label_sizes). labels: a vector with the shape of [label_len], all labels in the batch. label_sizes: a vector with the shape of [B], contains the length of each label in the batch.

- **in_seq_lens**

  A vector with the shape of [B]. The length of each input of the batch.

For details, please refer to _DataLoader/dataloader.py_
