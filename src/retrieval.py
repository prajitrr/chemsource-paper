import os
import pandas as pd
from chemsource import ChemSource
from pandarallel import pandarallel


def retrieve_text_synonyms_list(synonyms_list, model, source_priority="WIKIPEDIA"):
    """
    Retrieve text from a list of synonyms.
    
    Args:
        synonyms_list (list): A list of synonyms.
    
    Returns:
        tuple: A tuple containing the best synonym, the source of text, and the text itself.
    """
    
    results = []
    for synonym in synonyms_list:
        if isinstance(synonym, str):
            source, text = model.retrieve(synonym)
            if source == source_priority:
                return synonym, source, text
            results.append((synonym, source, text))

    results_filtered = [result for result in results if result[2] != "NO_RESULTS"]

    if results_filtered:
        return results_filtered[0]
    else:
        return None, None, None

def retrieve_text(harmonized_dataframe, ncbi_key=None, synonyms_column="SYNONYMS", source_column="SOURCE", updated_name_column="USED_NAME", text_column="TEXT", max_cores=8):
    """
    Retrieve text from the harmonized dataframe based on the specified columns.
    
    Args:
        harmonized_dataframe (pd.DataFrame): The harmonized dataframe.
        synonyms_column (str): The column containing synonyms.
        source_column (str): The column containing the source of the data.
        updated_name_column (str): The column containing the updated name.
        text_column (str): The column containing the text to be retrieved.
    
    Returns:
        pd.DataFrame: A DataFrame with the retrieved text.
    """
    harmonized_dataframe = harmonized_dataframe.copy()
    
    if ncbi_key:
        model = ChemSource(ncbi_key=ncbi_key)
        slow = False
    else:
        model = ChemSource()
        slow = True

    if slow:

        for index, row in harmonized_dataframe.iterrows():
            synonyms_list = row[synonyms_column]
            best_synonym, source, text = retrieve_text_synonyms_list(synonyms_list, model)
            harmonized_dataframe.loc[index, updated_name_column] = best_synonym
            harmonized_dataframe.loc[index, source_column] = source
            harmonized_dataframe.loc[index, text_column] = text

        return harmonized_dataframe

    else:
        pandarallel.initialize(progress_bar=True, nb_workers=min(min(8, os.cpu_count() - 1), max_cores))
        retrieval_results = harmonized_dataframe[synonyms_column].parallel_apply(
            lambda synonyms_list: retrieve_text_synonyms_list(synonyms_list, model)
        )

        harmonized_dataframe[updated_name_column] = retrieval_results.apply(lambda x: x[0])
        harmonized_dataframe[source_column] = retrieval_results.apply(lambda x: x[1])
        harmonized_dataframe[text_column] = retrieval_results.apply(lambda x: x[2])

        return harmonized_dataframe

def retrieve_public_data(input_dir, output_dir, ncbi_key=None):
    """
    Retrieve public data from the specified input directory save it to the output directory.

    Args:
        input_dir (str): The directory containing the input data files.
        output_file (str): The path to the output directory where the public data will be saved.
    """ 

    dataset_names = ["brain", "dust", "feces", "food", "iss", "mouse", "pcp", "plasma"]

    for dataset in dataset_names:
        input_file = os.path.join(input_dir, f"{dataset}_harmonized.parquet")
        harmonized_dataframe = pd.read_parquet(input_file)

        harmonized_dataframe = retrieve_text(harmonized_dataframe, ncbi_key=ncbi_key)
        harmonized_dataframe.to_parquet(os.path.join(output_dir, f"{dataset}_harmonized.parquet"), index=False)
        print(f"Processed {dataset} dataset and saved to {output_dir}")