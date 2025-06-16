import chemsource as cs

def classify_with_log_probs(name, body_text, model):
    try:
        classification_output = model.classify(name, body_text)
        return classification_output[0]
    except:
        return None
    