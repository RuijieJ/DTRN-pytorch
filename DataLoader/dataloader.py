import cv2
import numpy as np
import os, random

from Configs import param
norm_height = param.norm_height

class dataloader(object):
    def __init__(self, data_list_path, im_dir, char_list_path, batch_size=16, in_train=False):
        self.batch_size = batch_size
        self.in_train = in_train

        # ---- get characters dict
        self.char2num_dict = {}
        self.num2char_dict = {}
        f = open(char_list_path, 'r')
        chars = f.readline().rstrip()
        f.close()
        for i, char in enumerate(chars):
            self.char2num_dict[char] = i
            self.num2char_dict[i] = char

        # ---- get image names and ground truth
        f = open(data_list_path, 'r')
        lines = f.readlines()
        f.close()
        self.im_paths = []
        self.gts = []
        for line in lines:
            spl = line.split()
            self.im_paths.append(spl[0] + '.jpg')
            self.gts.append(''.join(spl[1:]).replace('<space>', ' '))
        self.im_paths = [os.path.join(im_dir, name) for name in self.im_paths]

        self.train_length = len(self.im_paths)
        self.batch_number = int(np.ceil(np.float(self.train_length) / np.float(batch_size)))

        self.data_id = 0  # count for data
        self.batch_id = 0 # count for batch
        self.epoch_id = 0 # count for epoch

        # ---- shuffle for the first epoch
        if in_train:
            self.shuffle()

    def __iter__(self):
        return self

    def __next__(self):
        one_batch = self.next_batch()
        self.batch_id += 1

        if self.batch_id >= self.batch_number:
            if self.in_train:
                self.shuffle()
            self.data_id = 0
            self.batch_id = 0
            self.epoch_id += 1
            raise StopIteration

        return one_batch

    # helper functions to generate sparse labels
    def _sparse_tuple_from(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.array(indices, dtype=dtype)
        values = np.array(values, dtype=dtype)
        shape = np.array([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=dtype)

        return indices.copy(), values.copy(), shape.copy()

    # helper functions to generate packed labels
    def _pack_form(self, sequences):
        total_seq_len = 0
        labels_sizes = []
        for i in range(len(sequences)):
            total_seq_len += len(sequences[i])
            labels_sizes.append(len(sequences[i]))
        packed_labels = np.zeros(total_seq_len, dtype=np.int32)
        labels_sizes = np.asarray(labels_sizes, dtype=np.int32)

        cur_pos = 0
        for i, seq in enumerate(sequences):
            packed_labels[cur_pos:cur_pos + labels_sizes[i]] = seq
            cur_pos += labels_sizes[i]

        return packed_labels, labels_sizes

    # helper function to get a data
    def get_item(self, idx):
        im_path = self.im_paths[idx]
        gt = self.gts[idx]

        # ---- read image, we do not do any augmentation here
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)  # [norm_height, Length]
        im = im.T  # [Length, norm_height] transpose because the batch input has a shape of [B, 1, Length, norm_height]
        im = np.float32((255 - im) / 255)  # normalize to 0~1
        seq_len = im.shape[0]  # get Lenght

        # ---- get label
        label = []
        for char in gt:
            label.append(self.char2num_dict[char])
        label = np.array(label, dtype=np.int32)
        label_len = len(gt)

        return im, label, seq_len, label_len, im_path, gt

    def next_batch(self):
        bucket_im = []
        bucket_label = []
        bucket_seq_len = []
        bucket_label_len =[]
        bucket_im_path = []
        bucket_gt_txt = []
        # ---- get images and labels (list form) for next batch
        for i in range(self.data_id, min(self.data_id+self.batch_size, self.train_length)):
            im, label, seq_len, label_len, im_path, gt_txt = self.get_item(i)
            bucket_im.append(im)
            bucket_label.append(label)
            bucket_seq_len.append(seq_len)
            bucket_label_len.append(label_len)
            bucket_im_path.append(im_path)
            bucket_gt_txt.append(gt_txt)

        self.data_id += self.batch_size

        # ---- get batch image (array form)
        batch_im = np.zeros([len(bucket_im), 1, max(bucket_seq_len), norm_height], dtype=np.float32) # [B, 1, Length, norm_height]
        for i in range(len(bucket_im)):
            batch_im[i, 0, :bucket_seq_len[i], :] = bucket_im[i]

        # ---- get batch label (array form)
        sparse_batch_label = self._sparse_tuple_from(bucket_label)
        packed_batch_label = self._pack_form(bucket_label)

        # ---- get batch seq_len (array form)
        batch_seq_len = np.array(bucket_seq_len)

        # ---- get total batch label len
        batch_label_len = sum(bucket_label_len)

        return batch_im, sparse_batch_label, packed_batch_label, \
               batch_seq_len, batch_label_len, bucket_im_path, bucket_gt_txt

    def shuffle(self):
        compact = list(zip(self.im_paths, self.gts))
        random.shuffle(compact)
        self.im_paths, self.gts = zip(*compact)


if __name__ == '__main__':
    dl = dataloader(
        data_list_path="/data/IAM/data/lang/lines/char/aachen/tr.txt",
        im_dir="/data/IAM/data/imgs/lines_h128/",
        char_list_path='alphabet_iam.txt',
        batch_size=16,
        in_train=True
    )
    a = 1
