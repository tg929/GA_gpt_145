#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script visualizes molecular properties from a file for a given population.
It reads a file containing SMILES, docking score, QED score, and SA score,
and generates two scatter plots to show the trade-offs between properties.
The docking score is negated so that higher values (better affinity) are on the right.
The plot title is dynamically generated to include the protein name from the input path.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path # Import the Path object for easy path manipulation

def visualize_scores(data_file: str, output_image: str, protein_name: str):
    """
    Loads data and creates the visualization plots.

    Args:
        data_file (str): Path to the input data file.
                         Format: SMILES\tDocking_Score\tQED\tSA
        output_image (str): Path for the output image file.
        protein_name (str): The name of the protein target for the plot title.
    """
    # --- 1. Load Data ---
    try:
        column_names = ['smiles', 'docking_score', 'qed_score', 'sa_score']
        df = pd.read_csv(data_file, sep='\t', header=None, names=column_names)
        print(f"Successfully loaded {len(df)} molecules from '{data_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found. Please check the path.", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}", file=sys.stderr)
        return

    # --- 2. Create a new column for the negative docking score ---
    df['neg_docking_score'] = -df['docking_score']

    # --- 3. Setup Plotting Style ---
    sns.set_theme(style="whitegrid")

    # --- 4. Create two subplots side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- 5. [MODIFIED] Create dynamic title ---
    # The title now includes the protein name passed to the function.
    title = f'Population Property Distribution of {protein_name}'
    fig.suptitle(title, fontsize=18, weight='bold')

    # --- 6. Plot 1: -Docking Score vs. QED Score ---
    sns.scatterplot(
        ax=axes[0],
        data=df,
        x='neg_docking_score',
        y='qed_score',
        alpha=0.7,
        edgecolor='w',
        s=50
    )
    axes[0].set_title('Binding Affinity vs. QED Score', fontsize=14)
    axes[0].set_xlabel('-Docking Score (Higher is Better)', fontsize=12)
    axes[0].set_ylabel('QED Score (Higher is Better)', fontsize=12)
    axes[0].grid(True)

    axes[0].tick_params(axis='y', direction='in')

    # --- 7. Plot 2: -Docking Score vs. SA Score ---
    sns.scatterplot(
        ax=axes[1],
        data=df,
        x='neg_docking_score',
        y='sa_score',
        alpha=0.7,
        edgecolor='w',
        s=50,
        color='coral'
    )
    axes[1].set_title('Binding Affinity vs. SA Score', fontsize=14)
    axes[1].set_xlabel('-Docking Score (Higher is Better)', fontsize=12)
    axes[1].set_ylabel('SA Score (Lower is Better)', fontsize=12)
    axes[1].grid(True)

    # --- 8. Adjust layout and save/show the image ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    try:
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to '{output_image}'.")
    except Exception as e:
        print(f"An error occurred while saving the image: {e}", file=sys.stderr)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate scatter plots for molecular properties from a data file.'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the input population data file. e.g., .../4r6e/generation_20/file.smi'
    )
    parser.add_argument(
        '--output_image',
        type=str,
        default='population_scores_english.png',
        help='Filename for the output image (default: population_scores_english.png)'
    )
    args = parser.parse_args()

    # --- [MODIFIED] Extract protein name from the input file path ---
    # This assumes the path structure is '.../protein_name/generation_xx/...'
    try:
        input_path = Path(args.input_file)
        protein_name = input_path.parent.parent.name
        print(f"Extracted protein name: {protein_name}")
    except IndexError:
        print("Warning: Could not automatically determine protein name from path. Using 'Unknown'.")
        protein_name = "Unknown"

    visualize_scores(args.input_file, args.output_image, protein_name)