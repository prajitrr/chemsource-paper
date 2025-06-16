def classify_with_log_probs(name, body_text, model):
    try:
        classification_output = model.classify(name, body_text)
        log_probs = []
        for i in classification_output[1].content:
            log_probs.append((i.token, i.logprob))
        return classification_output[0], log_probs
    except:
        return None