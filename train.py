from Configs import param
import random, numpy, torch, os

from DataLoader.dataloader import dataloader
from Nets.models import CNNLSTM, CNNDTRN
from Trainer.CTCTrainer import CTCTrainer

# ---- set random seed
torch.manual_seed(param.train.random_seed) 	    # cpu
torch.cuda.manual_seed(param.train.random_seed) # gpu
numpy.random.seed(param.train.random_seed) 	    # numpy
random.seed(param.train.random_seed) 		    # random and transforms
torch.backends.cudnn.deterministic=True 	    # cudnn

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow's print
	os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_id

	# ---- data loaders
	train_dataloader = dataloader(
        data_list_path=param.train.train_list,
        im_dir=param.image_dir,
        char_list_path=param.alphabet,
        batch_size=param.train.batch_size_tr,
        in_train=True
    )

	val_dataloader = dataloader(
		data_list_path=param.train.val_list,
		im_dir=param.image_dir,
		char_list_path=param.alphabet,
		batch_size=param.train.batch_size_va,
		in_train=False
	)

	# ---- model
	if param.model_name == 'CNNLSTM':
		param.net.dtrn = None
		model = CNNLSTM(param.net)
	elif param.model_name == 'CNNDTRN':
		param.net.rnn = None
		model = CNNDTRN(param.net)
	else:
		raise("Unknow model name '{}'. Must in ['CNNLSTM', 'CNNDTRN']"
			  .format(param.model_name))

	model = model.cuda()

	# ---- training
	trainer = CTCTrainer(
			model=model,
			train_dataloader=train_dataloader,
			val_dataloader=val_dataloader,
			param=param
		)
	trainer.train()
