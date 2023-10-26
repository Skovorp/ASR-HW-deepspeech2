import unittest

from tqdm import tqdm

from hw_asr.collate_fn.collate import collate_fn
from hw_asr.datasets import LibrispeechDataset
from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser

_TOTAL_ITERATIONS = 10
config_parser = ConfigParser.get_test_configs()
with clear_log_folder_after_use(config_parser):
    dataloaders = get_dataloaders(config_parser, config_parser.get_text_encoder())
    for part in ["train", "val"]:
        dl = dataloaders[part]
        for i, batch in tqdm(enumerate(iter(dl)), total=_TOTAL_ITERATIONS,
                                desc=f"Iterating over {part}"):
            if i >= _TOTAL_ITERATIONS: break