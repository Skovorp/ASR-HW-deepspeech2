import logging
from typing import List
import torch.nn.functional as F
import torch
logger = logging.getLogger(__name__)
import random


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
   
    # 'audio', 'spectrogram', 'duration', 'text', 'text_encoded', 'audio_path']


    max_spectrogram_len = max([x['spectrogram'].shape[2] for x in dataset_items])
    max_text_encoded_len = max([x['text_encoded'].shape[1] for x in dataset_items])

    outp = {
        'spectrogram': [],
        'text_encoded': [],
        'text_encoded_length': [],
        'text': [],
        'spectrogram_length': [],
        'audio_path': []
    }

    # in python -m unittest hw_asr/tests/test_dataloader.py prints shuffled lengths in batches both for train and val
    # weird
    # print("lens", [x['duration'] for x in dataset_items])
    # print(f"padding to spec {max_spectrogram_len}")

    for el in dataset_items:
        pad_spec = (0, max_spectrogram_len - el['spectrogram'].shape[2])
        pad_text = (0, max_text_encoded_len - el['text_encoded'].shape[1])

        outp['spectrogram'].append(F.pad(el['spectrogram'], pad_spec, 'constant', 0))
        outp['text_encoded'].append(F.pad(el['text_encoded'], pad_text, 'constant', 0))
        outp['text'].append(el['text'])
        outp['text_encoded_length'].append(el['text_encoded'].shape[1])
        outp['spectrogram_length'].append(el['spectrogram'].shape[2])
        outp['audio_path'].append(el['audio_path'])
    outp['audio_example'] = random.choice(dataset_items)['audio'].cpu()

    
    outp['spectrogram'] = torch.cat(outp['spectrogram'], 0)
    outp['text_encoded'] = torch.cat(outp['text_encoded'], 0)
    outp['text_encoded_length'] = torch.IntTensor(outp['text_encoded_length'])
    outp['spectrogram_length'] = torch.IntTensor(outp['spectrogram_length'])

    return outp


