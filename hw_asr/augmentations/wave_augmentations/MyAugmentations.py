import torch_audiomentations
from torch import Tensor
import torch
import torchaudio
import random

from hw_asr.augmentations.base import AugmentationBase
# from hw_asr.augmentations.Sequential import SequentialAugmentation


def random_from_to(min_, max_):
    return torch.rand(1).item() * (max_ - min_) + min_

def calc_energy_db(wav):
    return 10 * torch.log10((wav ** 2).mean())



class ChangeLoudness(AugmentationBase):
    def __init__(self, p, min_top, max_top, *args, **kwargs):
        self.min_top = min_top
        self.max_top = max_top
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() > self.p:
            return data
        wf_res = data / data.max()
        scale = random_from_to(self.min_top, self.max_top)
        return wf_res * scale


class AddNoise(AugmentationBase):
    def __init__(self, p, snr_min, snr_max, *args, **kwargs):
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.p = p
    
    def __call__(self, data: Tensor):
        if random.random() > self.p:
            return data
        snr = random_from_to(self.snr_min, self.snr_max)
        noise = torch.normal(0, 0.01, size=data.shape)
        db_change = calc_energy_db(data) - calc_energy_db(noise) - snr
        gain = (10 ** (db_change / 10)) ** 0.5
        return data + noise * gain


class LowPass(AugmentationBase):
    def __init__(self, p, freq_min, freq_max, *args, **kwargs):
        self.freq_min = torch.tensor(freq_min)
        self.freq_max = torch.tensor(freq_max)
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() > self.p:
            return data
        freq = torch.exp(random_from_to(torch.log(self.freq_min), torch.log(self.freq_max)))
        return torchaudio.functional.lowpass_biquad(data, 16000, freq, 10)


class HighPass(AugmentationBase):
    def __init__(self, p, freq_min, freq_max, *args, **kwargs):
        self.freq_min = torch.tensor(freq_min)
        self.freq_max = torch.tensor(freq_max)
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() > self.p:
            return data
        freq = torch.exp(random_from_to(torch.log(self.freq_min), torch.log(self.freq_max)))
        return torchaudio.functional.highpass_biquad(data, 16000, freq, 10)