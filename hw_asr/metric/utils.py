import editdistance

def calc_cer(target_text, predicted_text) -> float:
    if target_text == '':
        if predicted_text != '':
            return 1
        else:
            return 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if target_text == '':
        if predicted_text != '':
            return 1
        else:
            return 0
    target_tokens = target_text.strip().split(' ')
    predicted_tokens = predicted_text.strip().split(' ')
    return editdistance.eval(target_tokens, predicted_tokens) / len(target_tokens)