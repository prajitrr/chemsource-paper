import os
from ast import literal_eval

import pandas as pd
from preprocessing import filter_synonym_list, preprocess_chemical


def retrieve_public_data_file_paths(public_data_folder):
    """
    Retrieves the paths of public data files from the specified folder.

    Parameters:
    public_data_folder (str): Path to the folder containing public data files.

    Returns:
    tuple: A tuple containing dataframes for synonyms and detection frequencies.
    """

    for file in os.listdir(public_data_folder):
        if "synonyms" in file:
            public_synonyms_path = os.path.join(public_data_folder, file)
        elif "detection" in file:
            public_detection_frequencies_path = os.path.join(public_data_folder, file)
        else:
            raise ValueError(f"Unexpected file in public data folder: {file}")

    if public_synonyms_path and public_detection_frequencies_path:
        if "tsv" in public_synonyms_path:
            public_synonyms_df = pd.read_csv(public_synonyms_path, sep="\t")
        else:
            public_synonyms_df = pd.read_csv(public_synonyms_path)

        if "tsv" in public_detection_frequencies_path:
            public_detection_frequencies_df = pd.read_csv(
                public_detection_frequencies_path, sep="\t"
            )
        else:
            public_detection_frequencies_df = pd.read_csv(
                public_detection_frequencies_path
            )

        return public_synonyms_df, public_detection_frequencies_df

    return None, None

def harmonize_brain_data(brain_folder):
    """
    Harmonizes the ROSMAP brain dataset by reading and processing the files in the specified folder.

    Parameters:
    brain_folder (str): Path to the folder containing ROSMAP brain data files.

    Returns:
    pd.DataFrame: A DataFrame containing the harmonized data.
    """
    synonyms, detection_frequencies = retrieve_public_data_file_paths(brain_folder)

    corresponding_synonyms = []
    for index, row in detection_frequencies.iterrows():
        if row["compound_name"] != row["Compound_Name"]:
            raise ValueError("Compound name mismatch")
        compound_name = row["compound_name"]

        for _, synonym_row in synonyms.iterrows():
            if compound_name.casefold() in synonym_row["synonyms"].casefold():
                corresponding_synonyms.append(synonym_row["synonyms"])
                break
        else:
            corresponding_synonyms.append(None)

    detection_frequencies["synonyms"] = corresponding_synonyms
    detection_frequencies["synonyms"] = detection_frequencies["synonyms"].apply(
        literal_eval
    )

    detection_frequencies = detection_frequencies[
        ["featureID", "compound_name", "synonyms", "DF"]
    ]
    detection_frequencies = detection_frequencies.rename(
        columns={
            "DF": "DETECTION_FREQUENCY",
            "synonyms": "SYNONYMS",
            "compound_name": "COMPOUND_NAME",
            "featureID": "FEATURE_ID",
        }
    )

    detection_frequencies = detection_frequencies.groupby(
        "FEATURE_ID", as_index=False
    ).agg({"COMPOUND_NAME": "first", "SYNONYMS": "sum", "DETECTION_FREQUENCY": "first"})

    detection_frequencies["SYNONYMS"] = detection_frequencies["SYNONYMS"].apply(
        lambda x: list(dict.fromkeys(x))
    )
    detection_frequencies["DETECTION_FREQUENCY"] = detection_frequencies["DETECTION_FREQUENCY"].astype(float)
    return detection_frequencies


def harmonize_split_public_data(public_data_folder):
    """
        Harmonizes the split public dataset by reading and processing the files in the specified folder.

        Parameters:
        public_data_folder (str): Path to the folder containing public data files.
    s
        Returns:
        pd.DataFrame: A DataFrame containing the harmonized data.
    """
    synonyms, detection_frequencies = retrieve_public_data_file_paths(
        public_data_folder
    )

    synonyms = synonyms.dropna(subset=["X.Scan."])
    synonyms_dict = dict(
        zip(synonyms["X.Scan."].astype(int), synonyms["synonyms"].apply(literal_eval))
    )
    detection_frequencies["synonyms"] = detection_frequencies["featureID"].map(
        synonyms_dict
    )

    detection_frequencies = detection_frequencies[
        ["featureID", "compound_name", "synonyms", "DF"]
    ]
    detection_frequencies = detection_frequencies.rename(
        columns={
            "DF": "DETECTION_FREQUENCY",
            "synonyms": "SYNONYMS",
            "compound_name": "COMPOUND_NAME",
            "featureID": "FEATURE_ID",
        }
    )

    detection_frequencies["SYNONYMS"] = detection_frequencies["SYNONYMS"].apply(
        lambda x: filter_synonym_list(x) if isinstance(x, list) else None
    )
    detection_frequencies["SYNONYMS"] = detection_frequencies["SYNONYMS"].apply(
        lambda x: preprocess_chemical(x) if isinstance(x, list) else None
    )
    detection_frequencies["SYNONYMS"] = detection_frequencies["SYNONYMS"].apply(
        lambda x: x if isinstance(x, list) and len(x) > 0 else None
    )

    if detection_frequencies["SYNONYMS"].isnull().any():
        detection_frequencies.dropna(
            subset=["COMPOUND_NAME", "SYNONYMS"], inplace=True, how="all"
        )
        mask = detection_frequencies["SYNONYMS"].isnull()
        detection_frequencies.loc[mask, "SYNONYMS"] = detection_frequencies.loc[
            mask, "COMPOUND_NAME"
        ].apply(lambda x: [x])

    detection_frequencies = detection_frequencies.groupby(
        "FEATURE_ID", as_index=False
    ).agg({"COMPOUND_NAME": "first", "SYNONYMS": "sum", "DETECTION_FREQUENCY": "first"})

    detection_frequencies["SYNONYMS"] = detection_frequencies["SYNONYMS"].apply(
        lambda x: list(dict.fromkeys(x))
    )

    detection_frequencies["DETECTION_FREQUENCY"] = detection_frequencies["DETECTION_FREQUENCY"].astype(float)

    return detection_frequencies


def harmonize_combined_public_dataset(public_data_folder):
    """
    Harmonizes the combined public dataset by reading and processing the file in the specified folder.

    Parameters:
    public_data_folder (str): Path to the folder containing public data file.

    Returns:
    pd.DataFrame: A DataFrame containing the harmonized data.
    """
    data_files = os.listdir(public_data_folder)
    if len(data_files) != 1:
        raise ValueError("Expected exactly one file in the public data folder.")
    data_file_path = os.path.join(public_data_folder, data_files[0])
    if "tsv" in data_file_path:
        public_data_df = pd.read_csv(data_file_path, sep="\t")
    else:
        public_data_df = pd.read_csv(data_file_path)

    public_data_df = public_data_df[["featureID", "compound_name", "synonyms", "DF"]]
    public_data_df = public_data_df.rename(
        columns={
            "DF": "DETECTION_FREQUENCY",
            "synonyms": "SYNONYMS",
            "compound_name": "COMPOUND_NAME",
            "featureID": "FEATURE_ID",
        }
    )
    public_data_df["SYNONYMS"] = public_data_df["SYNONYMS"].apply(literal_eval)
    public_data_df["SYNONYMS"] = public_data_df["SYNONYMS"].apply(
        lambda x: filter_synonym_list(x) if isinstance(x, list) else None
    )
    public_data_df["SYNONYMS"] = public_data_df["SYNONYMS"].apply(
        lambda x: preprocess_chemical(x) if isinstance(x, list) else None
    )
    public_data_df["SYNONYMS"] = public_data_df["SYNONYMS"].apply(
        lambda x: x if isinstance(x, list) and len(x) > 0 else None
    )
    if public_data_df["SYNONYMS"].isnull().any():
        public_data_df.dropna(
            subset=["COMPOUND_NAME", "SYNONYMS"], inplace=True, how="all"
        )
        mask = public_data_df["SYNONYMS"].isnull()
        public_data_df.loc[mask, "SYNONYMS"] = public_data_df.loc[
            mask, "COMPOUND_NAME"
        ].apply(lambda x: [x])

    public_data_df = public_data_df.groupby("FEATURE_ID", as_index=False).agg(
        {"COMPOUND_NAME": "first", "SYNONYMS": "sum", "DETECTION_FREQUENCY": "first"}
    )

    public_data_df["SYNONYMS"] = public_data_df["SYNONYMS"].apply(
        lambda x: list(dict.fromkeys(x))
    )

    public_data_df["DETECTION_FREQUENCY"] = public_data_df["DETECTION_FREQUENCY"].astype(float)

    return public_data_df

def harmonize_all_public_data(public_data_folder, output_folder):
    """
    Harmonizes all public datasets by reading and processing the files in the specified folder.

    Parameters:
    public_data_folder (str): Path to the folder containing public data files.

    Returns:
    None
    """

    split_datasets = ["dust", "feces", "iss", "mouse", "plasma"]
    combined_datasets = ["food", "pcp"]

    all_datasets = split_datasets + combined_datasets + ["brain"]
    for dir in os.listdir(public_data_folder):
        dir_path = os.path.join(public_data_folder, dir)
        if os.path.isdir(dir_path):
            if "brain" in dir:
                harmonized_data = harmonize_brain_data(dir_path)
            elif any(item in dir for item in split_datasets):
                harmonized_data = harmonize_split_public_data(dir_path)
            elif any(item in dir for item in combined_datasets):
                harmonized_data = harmonize_combined_public_dataset(dir_path)
            else:
                raise ValueError(f"Unexpected directory: {dir}")

            dataset_name = [dataset for dataset in all_datasets if dataset in dir]

            if len(dataset_name) != 1:
                raise ValueError(f"Multiple or no dataset names found in {dir}")
            
            dataset_name = dataset_name[0]
            output_file = os.path.join(output_folder, f"{dataset_name}_harmonized.parquet")

            harmonized_data.to_parquet(output_file, index=False)
            print(f"Harmonized data for {dataset_name} saved to {output_file}")