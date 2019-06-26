import tensorflow as tf

from Configs import param
alphabet_path = param.alphabet

# get all characters
f = open(alphabet_path, 'r')
codec = f.readlines()[0]
codec_rev = {}
for i in range(0, len(codec)):
  codec_rev[codec[i]] = i
f.close()
dict_idx2char = dict((k, v) for v, k in codec_rev.items())

# log information
def LOG(log_info, log_filename, mode='a'):
    f = open(log_filename, mode=mode)
    print(log_info)
    f.write(log_info + '\r\n')
    f.close()

# decode from a sparse tensor
def get_row(sparse_tuple, row):
    optlist = []
    cnt = 0
    for pos in sparse_tuple[0]:
        if pos[0] == row:
            optlist.append(sparse_tuple[1][cnt])
        cnt += 1
    return optlist

# compute CER and display prediction
def compute_char_err(logits, seq_lens, sparse_labels, type='greedy', display=False, num_display=2):
    # ****** NOTE *******
    # we must reset the gragh in tensorflow,
    # or the program will slow down gradually due to the growth of computation gragh.
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    with tf.Session(config=config,) as sess:
        tf_logits = tf.convert_to_tensor(logits)
        tf_seq_lens = tf.convert_to_tensor(seq_lens)
        tf_sparse_labels = tf.SparseTensor(sparse_labels[0], sparse_labels[1], sparse_labels[2])
        if type == 'greedy':
            decoded, _ = tf.nn.ctc_greedy_decoder(tf_logits, tf.cast(tf_seq_lens, dtype='int32'))
        elif type == 'beam':
            decoded, _ = tf.nn.ctc_beam_search_decoder(tf_logits, tf.cast(tf_seq_lens, dtype='int32'), beam_width=5, merge_repeated=False)

        edit_distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), tf_sparse_labels, normalize=False)
        err = tf.reduce_sum(edit_distance)
        err, edit_distance = sess.run([err, edit_distance])

        char_preds = []
        if display:
            decoded_Array = sess.run(decoded[0])
            for i in range(num_display):
                pred = get_row(decoded_Array, i)
                char_pred = ''.join([dict_idx2char[idx] for idx in pred])
                char_preds.append(char_pred)

    return err, char_preds

