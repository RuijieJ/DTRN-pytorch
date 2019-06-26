import torch
from progressbar import *
from Utils.utils import *

class CTCTester(object):
    def __init__(self, model, test_dataloader, param):
        super(CTCTester, self).__init__()

        # ---- initialize
        self.p_te = param.test
        self.test_dataloader = test_dataloader
        self.model = model

    def load_state_dict(self):
        # ---- load model
        if os.path.exists(self.p_te.model_path):
            ckpt = torch.load(self.p_te.model_path)
            self.model.load_state_dict(ckpt['state_dict'])
            print('='*50)
            print("Test model '{}'".format(self.p_te.model_path))
        else:
            raise("Model '{}' not exist!".format(self.p_te.model_path))

    def test(self):
        # ---- load model
        self.load_state_dict()

        # ---- eval mode
        self.model.eval()

        char_errs = 0.
        label_lens = 0

        with torch.no_grad():

            widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
            progress = ProgressBar(widgets=widgets, maxval=10 * self.test_dataloader.batch_number).start()

            for step, data_batch in enumerate(self.test_dataloader):
                # ---- get data of a batch
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
                    type='beam'
                )
                # ---- sum the CER and label length
                char_errs += char_err
                label_lens += label_len

                progress.update(step * 10 + 1)
            progress.finish()

            # ---- compute average CER
            char_err_rate = char_errs / label_lens

        print('Test CER = {:.3f}%'.format(char_err_rate*100))
        print('='*50)

