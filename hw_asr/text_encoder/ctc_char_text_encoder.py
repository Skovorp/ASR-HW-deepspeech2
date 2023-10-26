from typing import List, NamedTuple
from collections import defaultdict

import torch
import math
from pyctcdecode import build_ctcdecoder
import numpy

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str = ''
    prev: str = ''
    logprob: float = 0


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm_path=None, lm_vocab_path=None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        if lm_path is not None and lm_vocab_path is not None:
            with open(lm_path, 'r') as f:
                fixed_text = f.read().lower()
            with open(lm_path, 'w') as f:
                f.write(fixed_text)

            with open(lm_vocab_path, 'r') as f:
                unigram_list = f.read()
                unigram_list = list(map(lambda x: x.lower().strip(), unigram_list.split('\n')))
            labels = [""] + list(self.alphabet)
            self.lm_decoder = build_ctcdecoder(
                labels,
                kenlm_model_path=lm_path,  
                alpha=0.5,
                beta=1.0,
                unigrams=unigram_list
            )

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        result = ''
        prev = -1
        for el in inds:
            if el == self.char2ind[self.EMPTY_TOK]:
                prev = self.char2ind[self.EMPTY_TOK]
            elif el != prev:
                prev = el
                result += self.ind2char[el]
        return result


    def calc_new_text(self, old_text: str, prev: str, new_el: str,) -> str:
        if new_el != prev and new_el != self.EMPTY_TOK:
            return old_text + new_el
        else:
            return old_text


    @staticmethod
    def truncate_hypos(hypos, max_len):
        seen = {}
        for h in hypos:
            if (h.text, h.prev) in seen:
                seen[(h.text, h.prev)] = math.log(math.exp(seen[(h.text, h.prev)]) + math.exp(h.logprob))
            else:
                seen[(h.text, h.prev)] = h.logprob
        joined = [Hypothesis(pair[0], pair[1], lp) for pair, lp in seen.items()]
        return sorted(joined, key=lambda x: x.logprob, reverse=True)[:max_len]


    def ctc_beam_search(self, probs: torch.tensor, probs_length: int,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char), f"voc_size ({voc_size}) != len(self.ind2char) ({len(self.ind2char)})"
        assert len(self.ind2char) < beam_size, f"My implementation skips extra sort by assuming beam_size is bigger then vocab. Make beam_size ({char_length}) bigger than {len(self.ind2char)}"
        hypos: List[Hypothesis] = []
        hypos.append(Hypothesis())
        log_probs = torch.log(probs)
        THRESHOLD = -4.6  # corresponds to a probability of about 1e-2
        
        
        for t in range(probs_length):
            hypo_groups = defaultdict(float)
            viable_indices = (log_probs[t] > THRESHOLD).nonzero(as_tuple=True)[0].tolist()
            for hypo in hypos:
                for ind in viable_indices:
                    new_log_prob = hypo.logprob + log_probs[t, ind].item()
                    new_text = self.calc_new_text(hypo.text, hypo.prev, self.ind2char[ind])
                    key = (new_text, self.ind2char[ind])
                    hypo_groups[key] += math.exp(new_log_prob)
            top_keys = sorted(hypo_groups, key=hypo_groups.get, reverse=True)[:beam_size]
            hypos = [Hypothesis(top_key[0], top_key[1], math.log(hypo_groups[top_key])) for top_key in top_keys]
        
        seen = {}
        for h in hypos:
            if h.text in seen:
                seen[h.text] = math.log(math.exp(seen[h.text]) + math.exp(h.logprob))
            else:
                seen[h.text] = h.logprob
        joined = [Hypothesis(text, '', lp) for text, lp in seen.items()]
        return sorted(joined, key=lambda x: x.logprob, reverse=True)


    def lm_beam_search(self, probs: torch.tensor, probs_length: int, 
                        beam_size: int=100):
        text = self.lm_decoder.decode(probs[:probs_length, :].numpy(), beam_width=beam_size)
        return Hypothesis(text, '', 1)