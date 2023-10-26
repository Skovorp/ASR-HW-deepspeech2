# Don't forget to support cases when target_text == ''
from jiwer import wer, cer

def calc_cer(target_text, predicted_text) -> float:
    return cer(target_text, predicted_text)



def calc_wer(target_text, predicted_text) -> float:
    return wer(target_text, predicted_text)