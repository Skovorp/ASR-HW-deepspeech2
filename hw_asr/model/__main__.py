import unittest

from tqdm import tqdm

from hw_asr.collate_fn.collate import collate_fn
from hw_asr.datasets import LibrispeechDataset
from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser

from hw_asr.model.my_model import MyModel




if __name__ == "__main__":
    print('hui')
    config_parser = ConfigParser.get_test_configs()
    with clear_log_folder_after_use(config_parser):
        dataloaders = get_dataloaders(config_parser, config_parser.get_text_encoder())
        dl = dataloaders['train']
    sample_batch = next(iter(dl))
    model = MyModel(128, 27, 800)
    print("outp shape", model(**sample_batch)['logits'].shape)


