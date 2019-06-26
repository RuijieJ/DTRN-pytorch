import pprint
from progressbar import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch_baidu_ctc import CTCLoss
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Utils.utils import *

class CTCTrainer(object):
	def __init__(self, model, train_dataloader, val_dataloader, param):
		super(CTCTrainer, self).__init__()

		# ---- create saving dir and backup files
		self.checkpoint_dir = param.saving_path
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		self._copy_backups()
		self.log_filename = os.path.join(self.checkpoint_dir, "log.txt")

		# ---- display configurations
		pp = pprint.PrettyPrinter(indent=4)
		LOG(pp.pformat(param), self.log_filename)
		LOG('='*50 + '\n# Params = {}'
		  	.format(sum(p.numel() for p in model.parameters() if p.requires_grad)),
			self.log_filename)
		self.p_tr = param.train

		# ---- construct data loaders
		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader

		# ---- construct model
		self.model = model

		# ---- loss function and optimizer
		self.num_classes = param.num_classes
		self.ctc = CTCLoss(reduction='mean', blank=self.num_classes-1)

		self.optimizer = optim.RMSprop(
			self.model.parameters(),
			lr=self.p_tr.learning_rate,
			alpha=0.95,
			weight_decay=self.p_tr.l2_weight_decay)
		self.scheduler = ReduceLROnPlateau(self.optimizer,
										   factor=self.p_tr.lr_factor,
										   patience=self.p_tr.lr_reduce_patient,
										   mode='min')

	def _copy_backups(self):
		# ---- backup files
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		os.system('cp -r * ' + os.path.join(self.checkpoint_dir, 'backup'))

	def train(self):
		# ---- main function of training
		try:
			self._train()
		except (Exception, KeyboardInterrupt) as e:
			print('\n---------- Catch a Exception while training -----------')
			raise e

	def load_state_dict(self):
		# ---- load continue model
		if self.p_tr.continue_train and os.path.exists(self.p_tr.continue_path):
			ckpt = torch.load(self.p_tr.continue_path)
			self.model.load_state_dict(ckpt['state_dict'])
			self.optimizer.load_state_dict(ckpt['optimizer'])
			LOG('Load Model from {}'.format(self.p_tr.continue_path), self.log_filename)

	def _train(self):
		# ---- load continue model if it exists
		self.load_state_dict()

		# ---- set flag that saves the best validation results
		self.min_cer_val = 1.

		# ---- train the network
		for epoch in range(self.p_tr.num_epochs):
			LOG('='*50, self.log_filename)
			LOG('-'*16 + 'Epoch {:02d} Training'.format(epoch + 1) + '-'*17, self.log_filename)

			# ---- train one epoch
			loss_train = self.epoch_train()
			LOG('[Epoch {}/{}] training loss = {:.3f}'
				.format(epoch+1, self.p_tr.num_epochs, loss_train),
				self.log_filename)

			# ---- validation one epoch
			LOG('-'*16 + 'Epoch {:02d} Validation'.format(epoch + 1) + '-'*15, self.log_filename)
			cer_val = self.epoch_val()

			self.scheduler.step(cer_val)

			# ---- saving model if validation CER decreases
			if cer_val < self.min_cer_val:
				self.min_cer_val = cer_val
				torch.save({'state_dict': self.model.state_dict(),
							'optimizer': self.optimizer.state_dict()},
					   		os.path.join(self.checkpoint_dir, 'm-epoch{}.pth.tar'.format(epoch+1)))
				LOG('Model saved as {}'
					.format(os.path.join(self.checkpoint_dir, 'm-epoch{}.pth.tar'.format(epoch+1))),
					self.log_filename)
			LOG('[Epoch {}/{}] val CER = {:.3f}%, beat val CER = {:.3f}%'
				.format(epoch + 1, self.p_tr.num_epochs, cer_val * 100, self.min_cer_val*100),
				self.log_filename)

	def epoch_train(self):
		self.model.train()

		train_loss = 0.

		widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
		progress = ProgressBar(widgets=widgets, maxval=10 * self.train_dataloader.batch_number).start()

		for step, data_batch in enumerate(self.train_dataloader):
			self.optimizer.zero_grad()

			# ---- get data of a batch
			images, sparse_labels, (packed_labels, label_sizes), \
			in_seq_lens, label_len, im_names, gt_txts\
				= data_batch
			# ---- set data type
			images = Variable(torch.from_numpy(images)).cuda()
			in_seq_lens = Variable(torch.Tensor(in_seq_lens)).cuda()
			packed_labels = Variable(torch.from_numpy(packed_labels)).cpu()
			label_sizes = Variable(torch.from_numpy(label_sizes)).cpu()
			packed_labels = packed_labels.requires_grad_(False)
			label_sizes = label_sizes.requires_grad_(False)

			# ---- forward
			logits, out_seq_lens = self.model(images, in_seq_lens)

			logits = logits.requires_grad_(True)
			logits = logits.transpose(0, 1).contiguous()	# [T, B, NUM_CLASSES]
			out_seq_lens = out_seq_lens.requires_grad_(False)

			# ---- compute loss
			loss = self.ctc(logits, packed_labels.cpu(), out_seq_lens.cpu(), label_sizes.cpu())

			# ---- backward
			loss.backward()
			if self.p_tr.clip:
				torch.nn.utils.clip_grad_norm_(
					filter(lambda p : p.requires_grad, self.model.parameters()),
					max_norm=self.p_tr.clip_norm,
					norm_type=2)
			self.optimizer.step()

			# ---- sum the loss
			train_loss += loss.item()

			# ---- display the prediction of the first image in the batch
			if self.p_tr.display_decoded and step % self.p_tr.disp_interval == 0:
				_, pred = compute_char_err(
					logits.cpu().detach().numpy(),
					out_seq_lens.cpu().detach().numpy(),
					sparse_labels,
					type='greedy',
					display=True,
					num_display=1
				)

				print('-'*50)
				print('[Step {}/{}] Loss = {}'
					  .format(step+1, self.train_dataloader.batch_number, loss.item()))
				print('Image: {}'.format(im_names[0]))
				print('Ground Truth: {}'.format(gt_txts[0]))
				print('Prediction : {}'.format(pred[0]))

			progress.update(10*step + 1)
		progress.finish()

		# ---- return average loss
		train_loss /= self.train_dataloader.batch_number
		return train_loss

	def epoch_val(self):

		self.model.eval()

		char_errs = 0.
		label_lens = 0

		with torch.no_grad():
			widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
			progress = ProgressBar(widgets=widgets, maxval=10 * self.val_dataloader.batch_number).start()
			for step, data_batch in enumerate(self.val_dataloader):
				# ---- get data
				images, sparse_labels, (_, _), \
				in_seq_lens, label_len, _, _ \
					= data_batch
				# ---- set data type
				images = torch.from_numpy(images).cuda()
				in_seq_lens = torch.Tensor(in_seq_lens).cuda()

				# ---- forward
				logits, out_seq_lens = self.model(images, in_seq_lens)
				logits = logits.transpose(0, 1).contiguous()

				# ---- compute CER (before normalized)
				char_err, _ = compute_char_err(
					logits.cpu().detach().numpy(),
					out_seq_lens.cpu().detach().numpy(),
					sparse_labels,
					type='greedy'
				)
				# ---- sum the CER and label length
				char_errs += char_err
				label_lens += label_len

				progress.update(step*10 + 1)
			progress.finish()

		# ---- compute average CER
		char_err_rate = char_errs / label_lens

		return char_err_rate