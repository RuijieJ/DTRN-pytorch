import os

from Configs import param
from DataLoader.dataloader import dataloader
from Nets.models import CNNLSTM, CNNDTRN
from Trainer.CTCTester import CTCTester

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow's print
    os.environ['CUDA_VISIBLE_DEVICES'] = param.gpu_id

    # ---- data loaders
    test_dataloader = dataloader(
        data_list_path=param.test.test_list,
        im_dir=param.image_dir,
        char_list_path=param.alphabet,
        batch_size=param.test.batch_size_te,
        in_train=False
    )

    # ---- model
    if param.model_name == 'CNNLSTM':
        model = CNNLSTM(param.net)
    elif param.model_name == 'CNNDTRN':
        model = CNNDTRN(param.net)
    else:
        raise ("Unknow model name '{}'. param.model_name must in ['CNNLSTM', 'CNNDTRN']"
               .format(param.model_name))

    model = model.cuda()

    # ---- test
    tester = CTCTester(model, test_dataloader, param)
    tester.test()
