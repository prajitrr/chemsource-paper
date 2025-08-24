import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from upsetplot import plot, from_indicators
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

METHODS = ["GPT_RAG", "DEEPSEEK_RAG", "GPT_NO_RAG", "SEARCH_GPT"]

def compute_metrics_single_method(harmonized_data, manual_data, method_column):
    harmonized_data = pd.get_dummies(harmonized_data[method_column].apply(pd.Series).stack()).groupby(level=0).sum()
    harmonized_data = harmonized_data[harmonized_data["INFO"] == 0]
    harmonized_data.drop(columns=["INFO"], inplace=True)

    manual_data = manual_data.loc[harmonized_data.index]

    harmonized_data = harmonized_data[["MEDICAL", "ENDOGENOUS", "FOOD", "PERSONAL CARE", "INDUSTRIAL"]]
    manual_data = manual_data[["MEDICAL", "ENDOGENOUS", "FOOD", "PERSONAL CARE", "INDUSTRIAL"]]
    sum = harmonized_data +  2 * manual_data

    # TN = 0, FP = 1, FN = 2, TP = 3

    val_counts = sum.apply(lambda x: x.value_counts()).fillna(0).astype(int)

    precisions = val_counts.apply(lambda x: x[3] / (x[1] + x[3]) if (x[1] + x[3]) > 0 else 0, axis=0).to_dict()
    recalls = val_counts.apply(lambda x: x[3] / (x[2] + x[3]) if (x[2] + x[3]) > 0 else 0, axis=0).to_dict()
    return {"precisions": precisions, "recalls": recalls}

def plot_metrics_single_method(metrics):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(metrics["precisions"].keys(), metrics["precisions"].values())
    ax[0].set_title("Precisions")
    ax[0].set_ylabel("Precision")
    ax[0].set_xlabel("Classes")
    ax[0].set_ylim(0, 1.05)

    ax[1].bar(metrics["recalls"].keys(), metrics["recalls"].values())
    ax[1].set_title("Recalls")
    ax[1].set_ylabel("Recall")
    ax[1].set_xlabel("Classes")
    ax[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

def plot_metrics_all_methods(harmonized_data, manual_data):
    
    categories = ["MEDICAL", "ENDOGENOUS", "FOOD", "PERSONAL CARE", "INDUSTRIAL"]
    x = np.arange(len(categories))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    width = 0.1

    for method in METHODS:
        metrics = compute_metrics_single_method(harmonized_data, manual_data, method)
        ax[0].bar(x, list(metrics["precisions"].values()), width, label=method)
        ax[1].bar(x, list(metrics["recalls"].values()), width, label=method)
        x = x + width

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(categories)

    ax[0].set_title("Precisions")
    ax[0].set_ylabel("Precision")
    ax[0].set_xlabel("Classes")
    ax[0].set_ylim(0, 1.05)

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(categories)

    ax[1].set_title("Recalls")
    ax[1].set_ylabel("Recall")
    ax[1].set_xlabel("Classes")
    ax[1].set_ylim(0, 1.05)

    ax[0].legend()
    ax[1].legend()


    plt.tight_layout()
    plt.show()

def compute_classification_count(harmonized_data, method_column):
    harmonized_data = pd.get_dummies(harmonized_data[method_column].apply(pd.Series).stack()).groupby(level=0).sum()
    return(len(harmonized_data[harmonized_data["INFO"] == 0]))

def compute_upsetplot_data(harmonized_data, method_column):
    harmonized_data_copy = harmonized_data.copy()
    harmonized_data_copy = pd.get_dummies(harmonized_data_copy[method_column].apply(pd.Series).stack()).groupby(level=0).sum()
    harmonized_data_copy = harmonized_data_copy[harmonized_data_copy["INFO"] == 0]
    harmonized_data_copy.drop(columns=["INFO"], inplace=True)
    harmonized_data_copy = harmonized_data_copy.astype(bool)
    return harmonized_data_copy

def plot_classification_counts(harmonized_data):
    counts = {}
    for method in METHODS:
        counts[method] = compute_classification_count(harmonized_data, method)
    plt.bar(counts.keys(), counts.values())
    plt.title("Classification Counts")
    plt.ylabel("Count")
    plt.xlabel("Methods")
    plt.show()

def plot_upsetplots_automated(harmonized_data, min_subset_size="0.5%"):
    methods = ["GPT_RAG", "DEEPSEEK_RAG", "GPT_NO_RAG", "SEARCH_GPT"]
    for method in methods:
        upset_data = compute_upsetplot_data(harmonized_data, method)
        fig = plot(
            from_indicators(upset_data),
            subset_size="count",
            show_counts=True,
            min_subset_size=min_subset_size,
            sort_by="cardinality",
        )
        plt.title(f"UpSet Plot for {method}")

def plot_upsetplot_manual(harmonized_data, min_subset_size="0.5%"):
    fig = plot(from_indicators(harmonized_data[["MEDICAL", "ENDOGENOUS", "FOOD", "PERSONAL CARE", "INDUSTRIAL"]].astype(bool)),
        subset_size="count",
        show_counts=True,
        min_subset_size=min_subset_size,
        sort_by="cardinality",
    )
    plt.title("UpSet Plot (Manual)")

def plot_sankey_diagram_1(df, save_path=None):
    links = pd.concat([
    df[['CLASSIFIED','SOURCE']].rename(columns={'CLASSIFIED':'source','SOURCE':'target'}),
    df[['SOURCE','CLASS']].rename(columns={'SOURCE':'source','CLASS':'target'})
    ])


    link_counts = links.value_counts().reset_index(name='value')


    labels = list(pd.unique(link_counts[['source','target']].values.ravel()))
    label_to_id = {label:i for i,label in enumerate(labels)}

    sources = link_counts['source'].map(label_to_id)
    targets = link_counts['target'].map(label_to_id)
    values = link_counts['value']
    hover_text = [
        f"{s} → {t}: {v:,}"
        for s, t, v in zip(link_counts['source'], link_counts['target'], values)
    ]

    fig = go.Figure(data=[go.Sankey(
    node=dict(
    pad=15,
    thickness=20,
    line=dict(color="black", width=0.5),
    label=labels,
    ),
    link=dict(
    source=sources,
    target=targets,
    value=values,
    customdata=hover_text,
    hovertemplate="%{customdata}<extra></extra>"
    ),
    valueformat=",.0f",  
    valuesuffix=""      
    )])


    fig.update_layout(
        title_text="Sankey Diagram",
        font_size=12,
        hoverlabel=dict(
            namelength=-1,  
            bgcolor="white",
            bordercolor="black",
            font_size=12
        ),
        separators=",." 
    )

    fig.update_traces(
        valueformat=",.0f",  
        valuesuffix=""   
    )

    fig.show()
    if save_path:
        fig.write_html(save_path)

def plot_sankey_diagram_2(df, save_path=None):
    links = pd.concat([
        df[['chemsource Class', 'Manual Class']].rename(columns={'chemsource Class': 'source', 'Manual Class': 'target'}),
        df[['Manual Class', 'Match Count']].rename(columns={'Manual Class': 'source', 'Match Count': 'target'})
    ])

    link_counts = links.value_counts().reset_index(name='value')

    labels = list(pd.unique(link_counts[['source', 'target']].values.ravel()))
    label_to_id = {label: i for i, label in enumerate(labels)}

    sources = link_counts['source'].map(label_to_id)
    targets = link_counts['target'].map(label_to_id)
    values = link_counts['value']

    hover_text = [
        f"{s} → {t}: {v:,}"
        for s, t, v in zip(link_counts['source'], link_counts['target'], values)
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>"
        ),
        
        valueformat=",.0f",  
        valuesuffix=""       
    )])

    fig.update_layout(
        title_text="Sankey Diagram: ChemSource vs Manual Classification",
        font_size=12,
        hoverlabel=dict(
            namelength=-1,  
            bgcolor="white",
            bordercolor="black",
            font_size=12
        ),
       
        separators=",."  
    )

    fig.update_traces(
        valueformat=",.0f", 
        valuesuffix=""       
    )

    fig.show()

    if save_path:
        fig.write_html(save_path)
