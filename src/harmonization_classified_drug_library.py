import pandas as pd

from ast import literal_eval

VALID_ITEMS = ["INDUSTRIAL", "PERSONAL CARE", "FOOD", "MEDICAL", "ENDOGENOUS", "INFO"]
CLASSIFIED_COLUMNS = ["INDUSTRIAL", "PERSONAL CARE", "FOOD", "MEDICAL", "ENDOGENOUS"]

def check_output_validity(input_list):
    return all(item in VALID_ITEMS for item in input_list)

def harmonize_manual_classification_list(manual_classification_list):
    manual_classification_list = manual_classification_list.strip().upper().replace("DRUG METABOLITE", "MEDICAL")
    output = manual_classification_list.strip().upper().split(",")
    output = [item.strip() for item in output]

    if not set(output).issubset(set(CLASSIFIED_COLUMNS)):
        raise ValueError(f"Invalid manual classification terms found: {set(output) - set(CLASSIFIED_COLUMNS)}")
    return output

def harmonize_automated_classification_list(automated_classification_list):
    automated_classification_list = literal_eval(automated_classification_list)
    automated_classifications = automated_classification_list[0]
    output = automated_classifications.strip().upper().split(",")
    output = [item.strip() for item in output]
    if not set(output).issubset(set(VALID_ITEMS)):
        raise ValueError(f"Invalid automated classification terms found: {set(output) - set(VALID_ITEMS)}")
    return output

def harmonize_search_classification_list(automated_classification_list):
    if ("Cymethion is a synonym" in automated_classification_list 
        or "Dichlorophene is utilized in" in automated_classification_list 
        or "Ginkgolide A is a terpenic" in automated_classification_list
        or "Dimetridazole is a synthetic nitroimidazole" in automated_classification_list
        or "P-nitrophenyl beta-D-glucopyranoside is primarily used" in automated_classification_list
        or "Flacitran is a synonym for luteolin" in automated_classification_list
        or "Glycerol 1-octadecyl ether, also known" in automated_classification_list
        or "Sulfuric acid, also known as dihydrogen sulfate" in automated_classification_list
        or "Methionine sulfoxide is an oxidation product of the amino" in automated_classification_list
        or "Xanthaurine, also known as quercetin, is a flavonoid" in automated_classification_list
        or ';' not in automated_classification_list):
        return ["INFO"]
    output = automated_classification_list.split(";")[0]
    output.strip("()' ")
    output = output.split(",")
    output = [item.strip("()' ") for item in output]

    if not set(output).issubset(set(VALID_ITEMS)):
        print(automated_classification_list)
        raise ValueError(f"Invalid automated classification terms found: {set(output) - set(VALID_ITEMS)}")
    return output

def harmonize_manual_classification(classified_drug_library_data_path):
    df = pd.read_csv(classified_drug_library_data_path)
    df["FEATURE_ID"] = df.index
    df = df.rename(columns={"manual_classification": "MANUAL_CLASSIFICATION"})

    df["MANUAL_CLASSIFICATION"] = df["MANUAL_CLASSIFICATION"].apply(harmonize_manual_classification_list)

    one_hot_encoded_items = pd.get_dummies(df['MANUAL_CLASSIFICATION'].apply(pd.Series).stack()).groupby(level=0).sum()

    df = pd.concat([df["FEATURE_ID"], one_hot_encoded_items], axis=1)
    return df

def harmonize_automated_classification(classified_drug_library_data_path):
    df = pd.read_csv(classified_drug_library_data_path)
    df["FEATURE_ID"] = df.index
    df = df.rename({"site": "SOURCE", "chemsource_output_deepseek-v3": "DEEPSEEK_RAG", "chemsource_output_gpt-4-1": "GPT_NO_RAG", "chemsource_output_gpt-4o": "GPT_RAG", "chemsource_output_search_gpt": "SEARCH_GPT"}, axis=1)
    df = df[["FEATURE_ID", "SOURCE", "DEEPSEEK_RAG", "GPT_NO_RAG", "GPT_RAG", "SEARCH_GPT"]]

    df["DEEPSEEK_RAG"] = df["DEEPSEEK_RAG"].apply(harmonize_automated_classification_list)
    df["GPT_NO_RAG"] = df["GPT_NO_RAG"].apply(harmonize_automated_classification_list)
    df["GPT_RAG"] = df["GPT_RAG"].apply(harmonize_automated_classification_list)
    df["SEARCH_GPT"] = df["SEARCH_GPT"].apply(harmonize_search_classification_list)

    return df

def retrieve_sankey_1_data(harmonized_drug_library_df, column="GPT_RAG"):
    df = harmonized_drug_library_df[["SOURCE", column]].copy()
    mask = df[column].apply(lambda x: "INFO" not in x)

    
    cut_len = len(df) - len(df[mask])
    df = df[mask]
    df["CLASSIFIED"] = "chemsource Classified"
    df["CLASS"] = df[column].apply(lambda x: "MEDICAL" if x==["MEDICAL"] else "Not MEDICAL")
    cols = ["CLASSIFIED", "SOURCE", "CLASS"]

    classified_data = ["Not Classified"] * (cut_len)
    source_data = ["N/A"] * ( cut_len)
    class_data = ["N/A "] * (cut_len)

    # create new df with same cols and 800 entries
    new_df = pd.DataFrame(columns=cols)

    new_df["CLASSIFIED"] = classified_data
    new_df["SOURCE"] = source_data
    new_df["CLASS"] = class_data

    df = pd.concat([df, new_df])
    df = df[["CLASSIFIED", "SOURCE", "CLASS"]]
    return df

def retrieve_sankey_2_data(harmonized_drug_library_df, harmonized_manual_df, column="GPT_RAG"):
    df = harmonized_drug_library_df[["FEATURE_ID", "SOURCE", column]].copy()

    automated_classifications = pd.get_dummies(harmonized_drug_library_df[column].apply(pd.Series).stack()).groupby(level=0).sum()
    automated_classifications = automated_classifications[automated_classifications["INFO"] != 1]
    
    automated_classifications.drop(columns=["INFO"], inplace=True)
    harmonized_manual_df.drop(columns=["FEATURE_ID"], inplace=True)
    harmonized_manual_df = harmonized_manual_df.loc[automated_classifications.index, :]

    is_medical_automated = automated_classifications.apply(lambda x: "MEDICAL" if x["MEDICAL"] == 1 and x.sum() <= 1 else "Not MEDICAL", axis =1)
    is_medical_manual = harmonized_manual_df.apply(lambda x: "MEDICAL " if x["MEDICAL"] == 1 and x.sum() <= 1 else "Not MEDICAL ", axis=1)


    # not_medical_automated = automated_classifications[(automated_classifications["MEDICAL"] == 0) |(automated_classifications.sum(axis=1) > 1)]
    # not_medical_automated = not_medical_automated.drop(columns=["MEDICAL"])
    # not_medical_automated = not_medical_automated[not_medical_automated.sum(axis=1) > 0]

    not_medical_manual = harmonized_manual_df[(harmonized_manual_df["MEDICAL"] == 0) |(harmonized_manual_df.sum(axis=1) > 1)]
    not_medical_manual = not_medical_manual.drop(columns=["MEDICAL"])
    not_medical_manual = not_medical_manual[not_medical_manual.sum(axis=1) > 0]

    shared_indices = automated_classifications.index.intersection(not_medical_manual.index)
    summed_classifications = automated_classifications.drop(columns=["MEDICAL"]).loc[shared_indices, :] + not_medical_manual.loc[shared_indices, :]
    summed_classifications = summed_classifications.map(lambda x: True if x==0 or x== 2 else False)
    row_sums = summed_classifications.sum(axis=1)

    matched_categories = row_sums.apply(lambda x: "Four categories" if x==4 else("Three categories" if x==3 else "Other"))

    final_output = pd.DataFrame(columns=["chemsource Class", "Manual Class", "Match Count"])
    final_output["chemsource Class"] = is_medical_automated
    final_output["Manual Class"] = is_medical_manual
    final_output["Match Count"] = matched_categories
    final_output["Match Count"] = final_output["Match Count"].where(final_output["chemsource Class"] == "Not MEDICAL", other="N/A")
    final_output.fillna({"Match Count": "N/A"}, inplace=True)

    return final_output