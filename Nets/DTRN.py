import torch
import torch.nn as nn

class DTRN(nn.Module):
	def __init__(self, input_size, dcg_hidden_size, trn_hidden_size,
				 discount_factor, dcg_dropout=0.0, trn_dropout=[0.]):
		super(DTRN, self).__init__()

		self._discount_factor = discount_factor
		self._dir_num = 2

		# ---- Dynamic Coefficients Generator: LSTM + FC
		# -- lstm
		self._add_dcg(
			input_size=input_size,
			hidden_size=dcg_hidden_size,
			dropout=dcg_dropout
		)
		# --fc
		self._coeff_linear = nn.ModuleDict()
		self._coeff_linear['fw_linear'] = nn.Linear(in_features=self._dcg_hidden_size,
													out_features=1)
		self._coeff_linear['bw_linear'] = nn.Linear(in_features=self._dcg_hidden_size,
													out_features=1)
		self._sigmoid = nn.Sigmoid()

		# ---- Temporal Residual Unit
		self._add_trn(
			input_size=input_size,
			hidden_size=trn_hidden_size,
			dropout=trn_dropout
		)

	def _add_dcg(self, input_size, hidden_size, dropout):
		self._dcg_type = type
		self._dcg_input_size = input_size
		self._dcg_hidden_size = hidden_size
		self._dcg_dropout = dropout
		self._dcg = nn.LSTM(input_size=input_size,
							hidden_size=hidden_size,
							num_layers=1,
							bidirectional=True,
							batch_first=True)
		self._dcg_dropout_layer = nn.Dropout(p=dropout)


	def _add_trn(self, input_size, hidden_size, dropout):
		# ---- construct our own RNN cells to change the behaviors in the time dimension
		self._trn_num_layers = len(hidden_size)
		self._trn_input_size = input_size
		self._trn_hidden_size = hidden_size
		self._trn_dropout = dropout

		self._trn_cells = nn.ModuleList()
		self._trn_dropout_layers = nn.ModuleList()
		n = self._dir_num
		for i in range(self._trn_num_layers):
			one_fw_cell = nn.LSTMCell(input_size=input_size if i == 0 else hidden_size[i - 1] * n,
									  hidden_size=hidden_size[i])

			one_bw_cell = nn.LSTMCell(input_size=input_size if i == 0 else hidden_size[i - 1] * n,
									  hidden_size=hidden_size[i])
			one_cell = nn.ModuleDict({'fw_cell': one_fw_cell, 'bw_cell': one_bw_cell})
			self._trn_cells.append(one_cell)

			one_dropout = nn.Dropout(p=dropout[i])
			self._trn_dropout_layers.append(one_dropout)

	def _coeff_generate(self, inputs):
		# ---- compute the dynamic weights
		# alpha_t = gamma * sigmoid(fc(input))
		dcg_output, _ = self._dcg(self._dcg_dropout_layer(inputs))
		fw_coeffs = self._discount_factor * self._sigmoid(
			self._coeff_linear['fw_linear'](dcg_output[:, :, :self._dcg_hidden_size]))  # (T, B, 1) or (B, T, 1)
		bw_coeffs = self._discount_factor * self._sigmoid(
			self._coeff_linear['bw_linear'](dcg_output[:, :, self._dcg_hidden_size:]))
		return fw_coeffs, bw_coeffs


	def	_trn_cell_forward(self, layer_index, inputs, coeffs):
		# ---- the computational flow of one TRN layer

		# -- obtain the forward/backward layer
		one_cell = self._trn_cells[layer_index]

		# -- x.shape = [B, T, N]
		max_seq_len = inputs.size(1)
		batch_size = inputs.size(0)
		hidden_size = self._trn_hidden_size[layer_index]

		# -- initialize h_0, c_0
		h_0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32, requires_grad=True)
		c_0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32, requires_grad=True)
		if inputs.is_cuda:
			h_0 = h_0.cuda()
			c_0 = c_0.cuda()

		# -- compute h_t
		# -- forward direction
		h_n_fw_list = list()
		for i in range(max_seq_len):
			# -- run LSTM for one step
			s0 = slice(0, None, None)
			s1 = slice(i, i + 1, None)
			s2 = slice(0, None, None)
			h_n, c_n = one_cell['fw_cell'](inputs[s0, s1, s2].squeeze(1), (h_0, c_0))

			# -- h_t = LSTM(input) + alpha_t * h_{t-1}
			if i > 0:
				h_n = h_n + coeffs['fw_coeffs'][s0, s1, s2].squeeze(-1) * h_0

			# -- update hidden state and cell state
			h_0, c_0 = (h_n, c_n)
			h_n_fw_list.append(h_n.unsqueeze(1))
		# -- obtain the whole output
		output_fw = torch.cat(h_n_fw_list, dim=1)

		# -- backward direction
		h_0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32, requires_grad=True)
		c_0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32, requires_grad=True)
		if inputs.is_cuda:
			h_0 = h_0.cuda()
			c_0 = c_0.cuda()

		h_n_bw_list = list()

		reverse_inputs = torch.flip(inputs, dims=(1, )) # reverse the input sequences

		for i in range(max_seq_len):
			# -- same with the forward direction
			s0 = slice(0, None, None)
			s1 = slice(i, i + 1, None)
			s2 = slice(0, None, None)
			h_n, c_n = one_cell['bw_cell'](reverse_inputs[s0, s1, s2].squeeze(1), (h_0, c_0))

			if i > 0:
				h_n = h_n + coeffs['bw_coeffs'][s0, s1, s2].squeeze(-1) * h_0

			h_0, c_0 = (h_n, c_n)
			h_n_bw_list.append(h_n.unsqueeze(1))

		output_bw = torch.cat(h_n_bw_list, dim=1)
		output_bw = torch.flip(output_bw, dims=(1, )) # reverse the output sequences

		# -- return both the forward and backward outputs
		return torch.cat([output_fw, output_bw], dim=-1)


	def forward(self, inputs):
		# -- run DCG and get corresponding dynamic weights
		fw_coeffs, bw_coeffs = self._coeff_generate(inputs)
		coeffs = {'fw_coeffs': fw_coeffs,
				  'bw_coeffs': torch.flip(bw_coeffs, dims=(1, ))} # reverse weights of the backward direction

		# -- run TRN
		trn_layer_input = inputs
		trn_layer_output = None

		for i in range(self._trn_num_layers):
			trn_dropout = self._trn_dropout_layers[i]
			trn_layer_output = self._trn_cell_forward(i, trn_dropout(trn_layer_input), coeffs)
			trn_layer_input = trn_layer_output

		# -- not return (h_n, c_n)
		return trn_layer_output, None

