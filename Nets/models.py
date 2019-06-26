from Nets.DTRN import *

class CNNLSTM(nn.Module):

	def __init__(self, param):   # param <== param.net
		super(CNNLSTM, self).__init__()
		self.param = param
		self.feature_map_h = param.cnn.in_dim

		# ---- construct CNN
		self.conv_layers = self._add_conv_layers(param, in_channels=1)

		# ---- construct LSTM
		self.rnn_input_size = param.cnn.conv_knum[-1] * self.feature_map_h
		self.rnn_input_dropout = nn.Dropout(p=param.rnn.dropout)
		self.rnn = nn.LSTM(
			input_size=self.rnn_input_size,
			hidden_size=param.rnn.hidden_size,
			num_layers=param.rnn.num_layers,
			dropout=param.rnn.dropout,
			bidirectional=True,
			batch_first=True)

		# ---- construct fully connected layer
		self.rnn_out_linear = nn.Linear(in_features=param.rnn.hidden_size * 2,
										out_features=param.rnn.out_dim)
		self.rnn_out_linear_dropout = nn.Dropout(p=param.rnn.linear_dropout)

		# ---- initialize RNN weights
		self._init_weights()

	def _init_weights(self):
		# only for RNN's parameters now
		nn.init.xavier_normal_(self.rnn.all_weights[0][0])
		nn.init.xavier_normal_(self.rnn.all_weights[0][1])
		nn.init.xavier_normal_(self.rnn.all_weights[1][0])
		nn.init.xavier_normal_(self.rnn.all_weights[1][1])

	def _add_conv_layers(self, param, in_channels=1):
		conv_knum = param.cnn.conv_knum
		conv_dropout = param.cnn.conv_dropout
		conv_ksize = param.cnn.conv_ksize
		conv_stride = param.cnn.conv_stride
		conv_padding = param.cnn.conv_padding
		conv_with_pool = param.cnn.conv_with_pool

		conv_layers = list()

		for i in range(len(conv_knum)):
			# ---- dropout
			if conv_dropout[i] > 0:
				conv_layers.append(nn.Dropout2d(p=conv_dropout[i]))

			# ---- conv layer
			conv_layers.append(
				nn.Conv2d(
					in_channels=conv_knum[i - 1] if i >= 1 else in_channels,
					out_channels=conv_knum[i],
					kernel_size=conv_ksize[i],
					stride=conv_stride[i],
					padding=conv_padding[i],
					bias=False
			))

			# ---- bn layer
			conv_layers.append(nn.BatchNorm2d(num_features=conv_knum[i]))

			# ---- activation
			conv_layers.append(nn.LeakyReLU())

			# ---- max pool
			if conv_with_pool[i]:
				self.feature_map_h = self.feature_map_h // 2
				conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers_cout = conv_knum[-1]
		return nn.Sequential(*conv_layers)

	def _compute_seq_len(self, seq_len):
		# ---- compute sequence lengths after CNN
		for i in range(len(self.param.cnn.conv_knum)):
			if self.param.cnn.conv_with_pool[i]:
				seq_len = seq_len // 2
		return seq_len.int()

	def forward(self, inputs, seq_len):
		# ---- cnn
		conv_feature_map = self.conv_layers(inputs)
		[batch_size, _, squeezed_max_seq_len, _] = conv_feature_map.size()
		conv_feature_map = conv_feature_map.permute(0, 2, 1, 3).reshape(batch_size, squeezed_max_seq_len, -1)

		# ---- rnn
		seq_len = self._compute_seq_len(seq_len)
		rnn_input = conv_feature_map
		rnn_out, _ = self.rnn(rnn_input)

		# ---- fc
		logits = self.rnn_out_linear(self.rnn_out_linear_dropout(rnn_out))

		return logits, seq_len.int()


class CNNDTRN(nn.Module):
	'''
	DCG is refer to the Dynamic Coefficients Generator (the secondary network)
	TRN is refer to the Temporal Residual Network (the primary network)
	'''
	def __init__(self, param):   # param <== param.net
		super(CNNDTRN, self).__init__()
		self.param = param

		# ---- construct CNN
		self.feature_map_h = param.cnn.in_dim
		self.conv_layers = self._add_conv_layers(param, in_channels=1)

		# ---- construct DTRN
		self.rnn_input_size = param.cnn.conv_knum[-1] * self.feature_map_h
		self.rnn = DTRN(
			input_size=self.rnn_input_size,
			dcg_hidden_size=param.dtrn.dcg_hidden_size,
			trn_hidden_size=param.dtrn.trn_hidden_size,
			discount_factor=param.dtrn.trn_discount_factor,
			dcg_dropout=param.dtrn.dcg_dropout,
			trn_dropout=param.dtrn.trn_dropout,
		)

		# ---- construct fc
		self.rnn_out_linear = nn.Linear(in_features=param.dtrn.trn_hidden_size[-1] * 2,
										out_features=param.dtrn.out_dim)
		self.rnn_out_linear_dropout = nn.Dropout(p=param.dtrn.linear_dropout)

	def _add_conv_layers(self, param, in_channels=1):
		conv_knum = param.cnn.conv_knum
		conv_dropout = param.cnn.conv_dropout
		conv_ksize = param.cnn.conv_ksize
		conv_stride = param.cnn.conv_stride
		conv_padding = param.cnn.conv_padding
		conv_with_pool = param.cnn.conv_with_pool

		conv_layers = list()

		for i in range(len(conv_knum)):
			# ---- dropout
			if conv_dropout[i] > 0:
				conv_layers.append(nn.Dropout2d(p=conv_dropout[i]))

			# ---- conv layer
			conv_layers.append(
				nn.Conv2d(
					in_channels=conv_knum[i - 1] if i >= 1 else in_channels,
					out_channels=conv_knum[i],
					kernel_size=conv_ksize[i],
					stride=conv_stride[i],
					padding=conv_padding[i],
					bias=False
			))

			# ---- bn layer
			conv_layers.append(nn.BatchNorm2d(num_features=conv_knum[i]))

			# ---- activation
			conv_layers.append(nn.LeakyReLU())

			# ---- max pool
			if conv_with_pool[i]:
				self.feature_map_h = self.feature_map_h // 2
				conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers_cout = conv_knum[-1]
		return nn.Sequential(*conv_layers)

	def _compute_seq_len(self, seq_len):
		# ---- compute sequence lengths after CNN
		for i in range(len(self.param.cnn.conv_knum)):
			if self.param.cnn.conv_with_pool[i]:
				seq_len = seq_len // 2
		return seq_len.int()

	def forward(self, inputs, seq_len):
		# ---- cnn
		conv_feature_map = self.conv_layers(inputs)
		[batch_size, _, squeezed_max_seq_len, _] = conv_feature_map.size()
		conv_feature_map = conv_feature_map.permute(0, 2, 1, 3).reshape(batch_size, squeezed_max_seq_len, -1)

		# ---- rnn
		seq_len = self._compute_seq_len(seq_len)
		rnn_input = conv_feature_map
		rnn_out, _ = self.rnn(rnn_input)

		# ---- fc
		logits = self.rnn_out_linear(self.rnn_out_linear_dropout(rnn_out))

		return logits, seq_len.int()
