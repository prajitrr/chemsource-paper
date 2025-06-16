import chemsource as cs

def chemsource_list_apply(synonyms, model):
    if not isinstance(model, cs.ChemSource):
        raise TypeError(f"Invalid model type. Model should be of type ChemSource but was {type(model)} instead.")
    chemsource_output = ""
    for item in synonyms:
        chemsource_output = model.chemsource(item)
        if chemsource_output[1][1] != "INFO" and chemsource_output[0][1] != '':
            return item, chemsource_output
    return item, chemsource_output
