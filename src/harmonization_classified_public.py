import pandas as pd

DATASET_NAMES = {"rosmap": "brain", 
                 "adrc": "feces",
                 "adrc_plasma": "plasma",
                 "dust": "dust",
                 "food": "food",
                 "iss": "iss",
                 "mouse": "mouse",
                 "personal": "pcp"
                 }

BIOSPECIMENS = ["brain", "feces", "plasma","food", "mouse"]
SYNTHETICS = ["iss", "pcp"]

VALID_ITEMS = ["INDUSTRIAL", "PERSONAL CARE", "FOOD", "MEDICAL", "ENDOGENOUS", "INFO"]
CLASSIFIED_COLUMNS = ["INDUSTRIAL", "PERSONAL CARE", "FOOD", "MEDICAL", "ENDOGENOUS"]
def check_output_validity(input_list):
    return all(item in VALID_ITEMS for item in input_list)

def harmonize_classified_public_data(classified_public_data_path):
    df = pd.read_csv(classified_public_data_path)
    df = df.drop(columns=["X.Scan.", "synonyms", "text", "name_used", "chemsource_output_gpt-4o","chemsource_output_gpt-4o_classprobs","site","synonyms_lower"])

    df.rename(columns={"DF": "DETECTION_FREQUENCY", "chemsource_output_gpt-4o_classification": "CLASSIFICATION", "dataset": "DATASET"}, inplace=True)
    df["CLASSIFICATION"] = df["CLASSIFICATION"].apply(lambda x: x.strip().split(',') if isinstance(x, str) else [])
    df["CLASSIFICATION"] = df["CLASSIFICATION"].apply(lambda x: [item.strip() for item in x])

    df["DATASET"] = df["DATASET"].apply(lambda x: DATASET_NAMES.get(x, x))

    # Check validity
    valid = df["CLASSIFICATION"].apply(check_output_validity)
    if not valid.all():
        raise ValueError("Invalid classification found in the dataset.")


    one_hot_encoded_items = pd.get_dummies(df['CLASSIFICATION'].apply(pd.Series).stack()).groupby(level=0).sum()

    df_encoded = pd.concat([one_hot_encoded_items, df["DETECTION_FREQUENCY"], df["DATASET"]], axis=1)

    df_encoded = df_encoded[df_encoded["INFO"] == 0]
    df_encoded.drop(columns=["INFO"], inplace=True)

    

    return df_encoded

def aggregate_public_data(harmonized_data):
    df_encoded = harmonized_data.copy()
    df_encoded[CLASSIFIED_COLUMNS] = df_encoded[CLASSIFIED_COLUMNS].multiply(df_encoded["DETECTION_FREQUENCY"], axis=0)
    df_encoded.drop(columns=["DETECTION_FREQUENCY"], inplace=True)
    df_dataset_groups = df_encoded.groupby("DATASET").aggregate('sum')
    df_dataset_groups = df_dataset_groups.div(df_dataset_groups.sum(axis=1), axis=0)
    return df_dataset_groups

def refinement_function(df_row):
    if df_row["DATASET"] in BIOSPECIMENS:
        if df_row["FOOD"] == 1 or df_row["ENDOGENOUS"] == 1:
            df_row[["INDUSTRIAL", "PERSONAL CARE", "MEDICAL"]] = 0
    elif df_row["DATASET"] in SYNTHETICS:
        if df_row["INDUSTRIAL"] == 1 or df_row["PERSONAL CARE"] == 1:
            df_row[["FOOD", "ENDOGENOUS", "MEDICAL"]] = 0
    return df_row

def aggregate_and_refine_public_data(harmonized_data):
    df_encoded = harmonized_data.copy()

    df_encoded = df_encoded.apply(refinement_function, axis=1)

    df_encoded[CLASSIFIED_COLUMNS] = df_encoded[CLASSIFIED_COLUMNS].multiply(df_encoded["DETECTION_FREQUENCY"], axis=0)
    df_encoded.drop(columns=["DETECTION_FREQUENCY"], inplace=True)
    df_dataset_groups = df_encoded.groupby("DATASET").aggregate('sum')
    df_dataset_groups = df_dataset_groups.div(df_dataset_groups.sum(axis=1), axis=0)
    return df_dataset_groups