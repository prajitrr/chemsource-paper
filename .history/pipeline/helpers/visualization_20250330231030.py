import pandas as pd
import numpy as np

def cs_output_to_upset(path_to_chemsource_dataframe, categories_col_name):
    if path_to_chemsource_dataframe.endswith(".csv"):
        data = pd.read_csv(path_to_chemsource_dataframe, index_col=0)
    elif path_to_chemsource_dataframe.endswith(".tsv"):
        data = pd.read_csv(path_to_chemsource_dataframe, sep="\t", index_col=0)
    else:
        raise ValueError("Invalid file format. Please provide a .csv or .tsv file.")

    data[categories_col_name] = data[categories_col_name].apply(lambda x: x.split(","))
    data[categories_col_name] = data[categories_col_name].apply(lambda x: [y.strip() for y in x])

    all_categories = set()
    for item in data[categories_col_name]:
        # print(item)
        for category in item:
            all_categories.add(category)
    
    all_categories = sorted(list(all_categories))
    # print(all_categories)
    one_hot_encoded = pd.DataFrame(data=None, index=data.index, columns=all_categories)
    for index, row in data.iterrows():
        for category in row[categories_col_name]:
            one_hot_encoded.loc[index, category] = 1
    
    one_hot_encoded.fillna(0, inplace=True)
    return one_hot_encoded

def evaluate_probs(log_prob_list):
    probability_dict = {} 
    category = ""
    probability = 0
    log_prob_list = [item[].replace("<｜end▁of▁sentence｜>", "").replace("")]

    for item in log_prob_list:
        key = item[0]
        if "," in key:
            category += key.split(",")[0]
            category = category.strip()
            category = category.replace(",", "")
            probability = float(np.exp(probability))
            probability_dict.update({category: probability})
            category = key.split(",")[1]
            probability = float(item[1])
        else:
            category += key
            probability += float(item[1])
    category = category.strip()
    category = category.replace(",", "")
    probability = float(np.exp(item[1]))
    probability_dict.update({category: probability})
    return probability_dict

def cs_output_to_upset_probs(chemsource_dataframe, categories_col_name, probs_col_name):
    data = chemsource_dataframe.copy()
    data[categories_col_name] = data[categories_col_name].apply(lambda x: x.split(","))
    data[categories_col_name] = data[categories_col_name].apply(lambda x: [y.strip() for y in x])

    all_categories = set()
    for item in data[categories_col_name]:
        for category in item:
            all_categories.add(category)
    
    all_categories = sorted(list(all_categories))
    data = pd.concat([data, pd.DataFrame(columns=all_categories)], axis=1)
    for index, row in data.iterrows():
        for category in row[probs_col_name].keys():
            data.loc[index, category] = row[probs_col_name][category]
    
    data.fillna(0, inplace=True)
    return data