import numpy as np
import matplotlib.pyplot as plt
from upsetplot import plot, from_indicators
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



CLASSIFIED_COLUMNS = ["INDUSTRIAL", "ENDOGENOUS", "FOOD", "PERSONAL CARE", "MEDICAL"]

def plot_stacked_bar_public(aggregated_df):
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(aggregated_df))
    for category in CLASSIFIED_COLUMNS:
        ax.bar(
            aggregated_df[category].index.tolist(),
            aggregated_df[category].tolist(),
            label=category,
            bottom=bottom
        )
        bottom += np.array(aggregated_df[category].tolist())
    plt.legend()

def plot_upsetplots_public(harmonized_data, min_subset_size="0.5%"):
    harmonized_data = harmonized_data.drop(columns=["DETECTION_FREQUENCY"])
    datasets = harmonized_data["DATASET"].unique()
    for dataset in datasets:
        dataset_data = harmonized_data[harmonized_data["DATASET"] == dataset]
        dataset_data.drop(columns=["DATASET"], inplace=True)
        upset_data = dataset_data.astype(bool)
        fig = plot(
            from_indicators(upset_data),
            subset_size="count",
            show_counts=True,
            min_subset_size=min_subset_size,
            sort_by="cardinality",
        )
        plt.title(f"UpSet Plot for {dataset}")