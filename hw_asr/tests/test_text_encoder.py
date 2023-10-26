import unittest

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import torch

class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder()

        test_tensor = torch.zeros(9, 28)
        # At t=0, '0' has the highest probability.
        test_tensor[0, [0, 1, 2]] = torch.FloatTensor([9, 7, 8])
        test_tensor[1, [0, 1, 2]] = torch.FloatTensor([9, 7, 8])
        test_tensor[2, [0, 1, 2]] = torch.FloatTensor([9, 10, 8])
        test_tensor[3, [0, 1, 2]] = torch.FloatTensor([9, 10, 8])
        test_tensor[4, [0, 1, 2]] = torch.FloatTensor([9, 10, 8])
        test_tensor[5, [0, 1, 2]] = torch.FloatTensor([9, 10, 8])
        test_tensor[6, [0, 1, 2]] = torch.FloatTensor([9, 10, 8])
        test_tensor[7, [7]] = torch.FloatTensor([100]) # should be ignored because prob_len
        test_tensor[8, [7]] = torch.FloatTensor([100]) # should be ignored because prob_len

        test_tensor = torch.softmax(test_tensor, 1)
        assert ((test_tensor.sum(1) - 1) == 0).all


        beam_size = 30
        prob_len = 7
        res = text_encoder.ctc_beam_search(test_tensor, prob_len, beam_size)
        # visualisation to find test cases
        # print(f"argmax: '{text_encoder.ctc_decode(test_tensor.argmax(dim=1).tolist())}'")
        # for beam in res[:5]:
        #     print(f"logp: {beam.logprob:.3f} text: '{beam.text}'")
        self.assertEqual(res[0].text, 'aa')
        self.assertEqual(res[1].text, 'ba')
        self.assertEqual(res[2].text, 'a') # this is argmax result

    # def test_lm_beam_search(self):
    #     text_encoder = CTCCharTextEncoder(alphabet=None, 
    #         lm_path="/home/ubuntu/asr_project_template/3-gram.pruned.3e-7.arpa", 
    #         lm_vocab_path='/home/ubuntu/asr_project_template/librispeech-vocab.txt')

    #     test_tensor = torch.rand(50, 28)
    #     test_tensor = torch.softmax(test_tensor, 1)
    #     assert ((test_tensor.sum(1) - 1) == 0).all

    #     prob_len = 45
    #     res = text_encoder.lm_beam_search(test_tensor, prob_len)
        
        