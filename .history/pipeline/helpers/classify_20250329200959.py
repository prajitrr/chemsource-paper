import chemsource as cs

def classify_with_log_probs(name, body_text, model):
    try:
        classification_output = model.classify(name, body_text)
        return classification_output[0]
    except:
        return None

    log_probs = []
    for i in classification_output[1].content:
        log_probs.append((i.token, i.logprob))
    

    all_data["classification_log_probs"][num] = log_probs