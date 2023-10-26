from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.utils.object_loading import get_dataloaders
import torchaudio

if __name__ == "__main__":
    my_wave_augs = [
        {"type": "AddNoise", "args": {"snr_min": 10, "snr_max": 25}},
        # {
        #     "type": "RandomApply",
        #     "args": {
        #         "augmentation": {"type": "HighPass", "args": {"freq_min": 10, "freq_max": 500}},
        #         "p": 0.5
        #     }
        # },
        {"type": "HighPass", "args": {"freq_min": 10, "freq_max": 500}},
        {"type": "LowPass", "args": {"freq_min": 1000, "freq_max": 8000}},
        {"type": "ChangeLoudness", "args": {"min_top": 0.5, "max_top": 1}},
    ]

    config_parser = ConfigParser.get_test_configs()
    config_parser._config['augmentations']['wave'] = my_wave_augs
    with clear_log_folder_after_use(config_parser):
        dataloaders = get_dataloaders(config_parser, config_parser.get_text_encoder())
    
    for i in range(10):
        torchaudio.save(
            f'/home/ubuntu/asr_project_template/hw_asr/augmentations/samples/sample_{i}.wav', 
            dataloaders['train'].dataset[i]['audio'], 16000)