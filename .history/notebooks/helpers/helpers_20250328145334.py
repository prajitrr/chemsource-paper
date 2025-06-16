import re
import chemsource as cs
import pandas as pd
import numpy as np

inchi_key_pattern = r"[A-Z]{14}-[A-Z]{10}-[A-Z]"
cas_pattern = r"^\d{2,7}-\d{2}-\d$"
acyl_amide_pattern = r"^[A-Z][a-z]{2}-C\d+:\d+$"
digit_string = r"\d+"
unknown_numerical_pattern = r"A^\d{2,10}-\d{3}-\d{2}-\d$"
unknown_databank_pattern = r"PD\d{6}"
unknown_databank_pattern_2 = r"SY\d{6}"
generic_databank_pattern = r"[A-Z]{1,5}\d+"
generic_databank_pattern_2 = r"[A-Z]{1,5}-\d+"

def preprocess_chemical(x):
    x = re.sub(r" from NIST14", "", x)
    x = re.sub(r"Spectral Match to ", "", x)
    x = re.sub(r"-unclear if this is accurate", "", x)
    x = re.sub(r"Putative ", "", x)
    x = re.sub(r"Massbank: ", "", x)
    x = re.sub(r"Massbank:PR\d+", "", x)
    x = re.sub(r"- [0-9][0-9].[0-9] eV", "", x)
    x = re.sub(r" cation", "", x)
    x = re.sub(r" anion", "", x)
    x = re.sub(r" in source fragment", "", x)
    x = re.sub(r"possibly - gamma-Valerobetaine see jones Nat Metabolism 2021", "gamma-Valerobetaine", x)
    x = re.sub(r"ReSpect:PM[0-9][0-9][0-9][0-9][0-9][0-9]", "", x)
    x = re.sub(r"^DL-", "", x)
    x = re.sub(r"^L-", "", x)
    x = re.sub(r"^D-", "", x)
    x = re.sub(r">=\d+% (LC/MS-UV)", "", x)
    x = re.sub(r"CollisionEnergy:\d+", "", x)
    x = x.strip()
    x = x.capitalize()
    x = x.replace(", (z)-", "")
    return x

def filter_synonym_list(synonym_list):
    synonym_list_copy = synonym_list.copy()
    for synonym in synonym_list:
        try:
            if "CHEMBL" in synonym:
                synonym_list_copy.remove(synonym)
            elif "UNII" in synonym:
                synonym_list_copy.remove(synonym)
            elif "DTXSID" in synonym:
                synonym_list_copy.remove(synonym)
            elif "CHEBI" in synonym:
                synonym_list_copy.remove(synonym)
            elif "HMS" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Spectral Match" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Tox21" in synonym:
                synonym_list_copy.remove(synonym)
            elif bool(re.match(cas_pattern, synonym)):
                synonym_list_copy.remove(synonym)
            elif bool(re.match(inchi_key_pattern, synonym)):
                synonym_list_copy.remove(synonym)
            elif bool(re.match(acyl_amide_pattern, synonym)):
                synonym_list_copy.remove(synonym)
            elif bool(re.match(digit_string, synonym)):
                synonym_list_copy.remove(synonym)
            elif bool(re.match(unknown_numerical_pattern, synonym)):
                synonym_list_copy.remove(synonym)
            elif bool(re.match(unknown_databank_pattern, synonym)):
                synonym_list_copy.remove(synonym)
            elif bool(re.match(unknown_databank_pattern_2, synonym)):
                synonym_list_copy.remove(synonym)
            elif bool(re.match(generic_databank_pattern, synonym)):
                synonym_list_copy.remove(synonym)
            elif bool(re.match(generic_databank_pattern_2, synonym)):
                synonym_list_copy.remove(synonym)
            elif "UniProt" in synonym:
                synonym_list_copy.remove(synonym)
            elif "SpecPlus" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Spectrum" in synonym:
                synonym_list_copy.remove(synonym)
            elif "BSPBio" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Bio1" in synonym:
                synonym_list_copy.remove(synonym)
            elif "MFCD" in synonym:
                synonym_list_copy.remove(synonym)
            elif "CBiol" in synonym:
                synonym_list_copy.remove(synonym)
            elif "BML3" in synonym:
                synonym_list_copy.remove(synonym)
            elif "CAS" in synonym:
                synonym_list_copy.remove(synonym)
            elif "InChI" in synonym:
                synonym_list_copy.remove(synonym)
            elif "MassBank" in synonym:
                synonym_list_copy.remove(synonym)
            elif "AKOS" in synonym:
                synonym_list_copy.remove(synonym)
            elif "NCGC" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Acon1" in synonym:
                synonym_list_copy.remove(synonym)
            elif "ACon1" in synonym:
                synonym_list_copy.remove(synonym)
            elif "MEGxp0" in synonym:
                synonym_list_copy.remove(synonym)
            elif synonym == "":
                synonym_list_copy.remove(synonym)
            elif "SPBio" in synonym:
                synonym_list_copy.remove(synonym)
            elif "KBio3" in synonym:
                synonym_list_copy.remove(synonym)
            elif "DivK1c" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Lopac0" in synonym:
                synonym_list_copy.remove(synonym)
            elif "KBioSS" in synonym:
                synonym_list_copy.remove(synonym)
            elif "NSC" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Compound NP-" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Compound NP" in synonym:
                synonym_list_copy.remove(synonym)
            elif "DGTS" in synonym:
                synonym_list_copy.remove(synonym)
            elif "KBio1" in synonym:
                synonym_list_copy.remove(synonym)
            elif "BRD" in synonym:
                synonym_list_copy.remove(synonym)
            elif "BRN" in synonym:
                synonym_list_copy.remove(synonym)
            elif "LMFA" in synonym:
                synonym_list_copy.remove(synonym)
            elif "HY-" in synonym:
                synonym_list_copy.remove(synonym)
            elif "MEGxm0" in synonym:
                synonym_list_copy.remove(synonym)
            elif "MEGx" in synonym:
                synonym_list_copy.remove(synonym)
            elif "ACon" in synonym:
                synonym_list_copy.remove(synonym)
            elif "BRD-" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Prestwick" in synonym:
                synonym_list_copy.remove(synonym)
            elif "MEGxp" in synonym:
                synonym_list_copy.remove(synonym)
            elif "MLS" in synonym:
                synonym_list_copy.remove(synonym)
            elif "EXP" in synonym:
                synonym_list_copy.remove(synonym)
            elif "DUP" in synonym:
                synonym_list_copy.remove(synonym)
            elif "AR-" in synonym:
                synonym_list_copy.remove(synonym)
            elif "Tocris-" in synonym:
                synonym_list_copy.remove(synonym)
            elif "CCRIS" in synonym:
                synonym_list_copy.remove(synonym)
            elif "; [M+H]+ C" in synonym:
                synonym_list_copy.remove(synonym)

        except:
            pass

    return synonym_list_copy

def preprocessing_function_synonyms(synonym_list):
    new_synonym_list = []
    for x in synonym_list:
        x = re.sub(r" from NIST14", "", x)
        x = re.sub(r"Spectral Match to ", "", x)
        x = re.sub(r"-unclear if this is accurate", "", x)
        x = re.sub(r"Putative ", "", x)
        x = re.sub(r"Massbank: ", "", x)
        x = re.sub(r"Massbank:PR\d+", "", x)
        x = re.sub(r"- [0-9][0-9].[0-9] eV", "", x)
        x = re.sub(r" cation", "", x)
        x = re.sub(r" anion", "", x)
        x = re.sub(r" in source fragment", "", x)
        x = re.sub(r"possibly - gamma-Valerobetaine see jones Nat Metabolism 2021", "gamma-Valerobetaine", x)
        x = re.sub(r"ReSpect:PM[0-9][0-9][0-9][0-9][0-9][0-9]", "", x)
        x = re.sub(r"^DL-", "", x)
        x = re.sub(r"^LD-", "", x)
        x = re.sub(r"^L-", "", x)
        x = re.sub(r"^D-", "", x)
        x = re.sub(r"^dl-", "", x)
        x = re.sub(r"^ld-", "", x)
        x = re.sub(r"^l-", "", x)
        x = re.sub(r"^d-", "", x)
        x = re.sub(r"(-)-", "", x)
        x = re.sub(r"(\+)-", "", x)
        x = re.sub(r">=\d+% (LC/MS-UV)", "", x)
        x = re.sub(r"CollisionEnergy:\d+", "", x)
        x = x.strip()
        x = x.capitalize()
        x = x.replace(", (z)-", "")
        new_synonym_list.append(x)
    new_synonym_list = list(dict.fromkeys(new_synonym_list))
    return new_synonym_list


def filter_chemical(x):
    filtered_chemical = filter_synonym_list([x])
    if len(filtered_chemical) == 0:
        return None
    else:
        return filtered_chemical[0]
    
def get_length(x):
    try:
        return len(x)
    except:
        return 0

def remove_n(python_list, n):
    try:
        python_list.pop(n)
    except:
        return []
    return python_list

def chemsource_list_apply(synonyms, model):
    if not isinstance(model, cs.ChemSource):
        raise TypeError(f"Invalid model type. Model should be of type ChemSource but was {type(model)} instead.")
    chemsource_output = ""
    for item in synonyms:
        chemsource_output = model.chemsource(item)
        if chemsource_output[1][1] != "INFO" and chemsource_output[0][1] != '':
            return item, chemsource_output
    return item, chemsource_output

def cs_output_to_upset(path_to_chemsource_dataframe):
    if path_to_chemsource_dataframe.endswith(".csv"):
        data = pd.read_csv(path_to_chemsource_dataframe, index_col=0)
    elif path_to_chemsource_dataframe.endswith(".tsv"):
        data = pd.read_csv(path_to_chemsource_dataframe, sep="\t", index_col=0)
    else:
        raise ValueError("Invalid file format. Please provide a .csv or .tsv file.")

    data["classification"] = data["classification"].apply(lambda x: x.split(","))
    data["classification"] = data["classification"].apply(lambda x: [y.strip() for y in x])

    all_categories = set()
    for item in data["classification"]:
        # print(item)
        for category in item:
            all_categories.add(category)
    
    all_categories = sorted(list(all_categories))
    # print(all_categories)
    one_hot_encoded = pd.DataFrame(data=None, index=data.index, columns=all_categories)
    for index, row in data.iterrows():
        for category in row["classification"]:
            one_hot_encoded.loc[index, category] = 1
    
    one_hot_encoded.fillna(0, inplace=True)
    return one_hot_encoded

def evaluate_probs(log_prob_list):
    probability_dict = {} 
    category = ""
    probability = 0
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
