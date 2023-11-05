import editdistance

def calc_cer(target_text: str, predicted_text: str) -> float:
    return editdistance.eval(target_text, predicted_text) / len(target_text)

def calc_wer(target_text: str, predicted_text: str) -> float:
    if target_text == '':
        if predicted_text != '':
            return 1.0
        else:
            return 0.0
    target_words = target_text.split(' ')
    predicted_words = predicted_text.split(' ')
    return editdistance.eval(target_words, predicted_words) / len(target_words)