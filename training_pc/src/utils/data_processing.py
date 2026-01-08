"""
Data preprocessing pipeline
uses data class from data_handling.py for cleaning
and normalizing data
"""

import os
from importlib.resources.readers import remove_duplicates
import pandas as pd
from pathlib import Path
from data_handling import Data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def preprocessing_scan_data(
        input_filepath,
        output_dir="../data/processed",
        fix_indices=True,
        handle_outliers="clip",
        outlier_method="iqr",
        outlier_columns=None,
        normalize_methods=None,
        normalize_per_scan=False,
        create_visualizations=True,
        verbose=True
):
    """
    Complete preprocessing workflow for scan data

    :param input_filepath: path for csv input
    :param output_dir: directory for output
    :param fix_indices: correcting index column
    :param handle_outliers: "clip" or None
    :param outlier_method: "iqr" or "zscore"
    :param outlier_columns: list of outliers
    :param normalize_methods: list methods for normalizing
    :param normalize_per_scan: normalize per scan instead of global
    :param create_visualizations: create visuals
    :param verbose: showing progress

    Returns:
        dict with processed data objects
    """

    if verbose:
        print("=" * 60)
        print("DATA PREPROCESSING PIPELINE")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ==============================================
    # 1. LOADING DATA
    # ==============================================

    if verbose:
        print("\n[1/6] Loading Data...")

    data = Data()
    data.load_from_csv(input_filepath, has_header=True)

    if verbose:
        info = data.get_info()
        print(f" Shape: {info['shape']}")
        print(f" Columns: {info["columns"]}")
        if "num_scans" in info:
            print(f" Scan Count: {info["num_scans"]}")
            print(f" Scans: {info["scans"]}")


    if verbose:
        print("\n[2/6] Checking data quality...")

    if "scan_id" in data.df.columns:
        duplicates = data.check_duplicates(subset=["scan_id", "angle"])
    else:
        duplicates = data.check_duplicates()

    if verbose:
        if duplicates > 0:
            print(f" Duplicates found: {duplicates}")
        else:
            print(" No duplicates")


    #data cleaning
    if verbose:
        print("\n[3/6] Cleaning data...")

    if remove_duplicates and duplicates > 0:
        if "scan_id" in data.df.columns:
            data.remove_duplicates(subset=["scan_id", "angle"])
        else:
            data.remove_duplicates()
        if verbose:
            print(f" {duplicates} duplicates removed")

    # correcting indices
    if fix_indices and "scan_id" in data.df.columns:
        data.fix_index_column()
        if verbose:
            print(" Index column corrected")

    # handling outliers
    if handle_outliers and outlier_columns:
        for col in outlier_columns:
            if col in data.df.columns:
                outlier_info = data.detect_outliers_iqr(col) if outlier_method == "iqr" \
                            else data.detect_outliers_zscore(col)

                if verbose and outlier_info['count'] > 0:
                    print(f"  Column '{col}': {outlier_info['count']} Ausreißer "
                          f"({outlier_info['percentage']:.2f}%)")

                if handle_outliers == 'clip':
                    data.clip_outliers(col, method=outlier_method)
                    if verbose:
                        print(f" Clipped")
                elif handle_outliers == 'remove':
                    data.remove_outliers(col, method=outlier_method)
                    if verbose:
                        print(f" Removed")


    if verbose:
        print("\n[4/6] Applying domain specific transformation...")

    data_centered = data.copy()
    if "angle" in data_centered.df.columns and "distance" in data_centered.df.columns:
        data_centered.correct_center_and_normalize()
        data_centered.save_to_csv(output_path / "data_centered_normalized.csv")
        if verbose:
            print(" Geometric centre correction applied")
            print(f" Saved: {output_path / "data_centered_normalized.csv"}")


    # ML - Standardizing

    results = {
        "original": data,
        "centered": data_centered if "angle" in data.df.columns else None
    }

    if normalize_methods:
        if verbose:
            print("\n[5/6] Applying ML-normalizations...")


        columns_to_normalize = []
        if "distance" in data.df.columns:
            columns_to_normalize.append("distance")


        for method in normalize_methods:
            data_normalized = data.copy()
            data_normalized.normalize(
                columns=columns_to_normalize,
                method=method,
                per_scan=normalize_per_scan
            )

            suffix = "_per_scan" if normalize_per_scan else "-global"
            filename = f"data_{method}{suffix}.csv"
            data_normalized.save_to_csv(output_path / filename)

            results[f"{method}{suffix}"] = data_normalized

            if verbose:
                print(f" {method.upper()}-normalization "
                      f"({"per scan" if normalize_per_scan else "global"}")
                print(f" Saved: {output_path / filename}")

    if create_visualizations:
        if verbose:
            print("\n[6/6] Erstelle Visualisierungen...")

        create_comparison_plots(results, output_path, verbose)
        create_advanced_visualizations(results, output_path)

    if verbose:
        print("\n" + "=" * 60)
        print("PREPROCESSING ABGESCHLOSSEN")
        print("=" * 60)
        print(f"\nAusgabeverzeichnis: {output_path}")
        print(f"Anzahl Dateien: {len(list(output_path.glob('*.csv')))}")

    return results


# python
def create_advanced_visualizations(results, output_dir):
    """
    Erstellt fortgeschrittene Visualisierungen für Scan-Daten.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    original_df = results['original'].df.copy()
    processed_df = results.get('centered', results['original']).df.copy()

    if 'normalized' not in processed_df.columns and 'distance_corrected' in processed_df.columns:
        mean_dist_corr = processed_df['distance_corrected'].mean()
        processed_df['normalized'] = (processed_df['distance_corrected'] / mean_dist_corr) if mean_dist_corr != 0 else 0
    elif 'normalized' not in processed_df.columns and 'distance' in processed_df.columns:
        mean_dist = processed_df['distance'].mean()
        processed_df['normalized'] = (processed_df['distance'] / mean_dist) if mean_dist != 0 else 0

    print(f"--- Erstelle Visualisierungen in {output_path} ---")

    # ---------------------------------------------------------
    # 1. Polar Plot (plot all scans, capped)
    # ---------------------------------------------------------
    if 'angle' in processed_df.columns and 'distance' in processed_df.columns and 'scan_id' in processed_df.columns:
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')

        unique_ids = processed_df['scan_id'].unique()
        max_plots = 12  # safe cap to avoid huge overcrowding
        ids_to_plot = unique_ids[:max_plots]

        colors = plt.cm.viridis(np.linspace(0, 1, len(ids_to_plot)))
        for i, sid in enumerate(ids_to_plot):
            subset = processed_df[processed_df['scan_id'] == sid]
            if subset.empty:
                continue
            rads = np.radians(subset['angle'])
            ax.scatter(rads, subset['distance'], color=colors[i], alpha=0.7, s=20, label=f'Scan {sid}')

        if len(unique_ids) > max_plots:
            ax.set_title(f"Polar-Ansicht: Räumliche Verteilung (erste {max_plots} von {len(unique_ids)} Scans)", va='bottom')
        else:
            ax.set_title("Polar-Ansicht: Räumliche Verteilung (alle Scans)", va='bottom')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        plt.savefig(output_path / 'polar_spatial_view.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  - Polar Plot erstellt.")
    else:
        print("  - Polar Plot übersprungen: 'angle' oder 'distance' oder 'scan_id' Spalte fehlt.")

    # ---------------------------------------------------------
    # 2. Boxplots: Distanzen vor und nach Bereinigung/Normalisierung
    # ---------------------------------------------------------
    plot_df = pd.DataFrame({
        'Original Distanz': original_df['distance'],
        'Normalisierte Distanz': processed_df['normalized']
    })

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_df, palette='viridis')
    plt.title('Vergleich der Distanzverteilung (Original vs. Normalisiert)')
    plt.ylabel('Distanzwert')
    plt.savefig(output_path / 'boxplot_distance_comparison.png', dpi=300)
    plt.close()
    print("  - Boxplot Vergleich erstellt.")

    # ---------------------------------------------------------
    # 3. Seaborn PairPlot (Merkmalsbeziehungen)
    # ---------------------------------------------------------
    numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
    valid_cols = [col for col in numeric_cols if processed_df[col].nunique() > 1]

    if len(valid_cols) >= 2:
        g = sns.pairplot(processed_df[valid_cols], diag_kind='kde')
        g.fig.suptitle("PairPlot: Beziehungen zwischen den verarbeiteten Merkmalen", y=1.02)
        plt.savefig(output_path / 'pairplot_features.png', dpi=300)
        plt.close()
        print("  - PairPlot erstellt.")
    else:
        print(f"  - PairPlot übersprungen: Nicht genügend numerische Spalten mit Varianz ({valid_cols}).")

    # ---------------------------------------------------------
    # 4. Korrelations-Heatmap
    # ---------------------------------------------------------
    if len(valid_cols) >= 2:
        plt.figure(figsize=(10, 8))
        corr_matrix = processed_df[valid_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Korrelationsmatrix der verarbeiteten Merkmale')
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300)
        plt.close()
        print("  - Korrelations-Heatmap erstellt.")
    else:
        print("  - Korrelations-Heatmap übersprungen: Nicht genügend numerische Spalten mit Varianz.")

    # ---------------------------------------------------------
    # 5. Interaktive Plotly-Visualisierung (HTML)
    # ---------------------------------------------------------
    if 'angle' in processed_df.columns and 'distance' in processed_df.columns:
        fig = px.scatter(
            processed_df,
            x="angle",
            y="distance",
            color="scan_id",
            hover_data=['distance_corrected', 'normalized'],
            title="Interaktive Analyse aller Scans (Zoombar)"
        )
        fig.write_html(output_path / 'interactive_analysis.html', include_plotlyjs='cdn')
        print("  - Interaktiver Plotly-HTML-Bericht erstellt.")
    else:
        print("  - Interaktiver Plotly-Plot übersprungen: 'angle' oder 'distance' Spalte fehlt.")

    print("--- Alle Visualisierungen abgeschlossen. ---")


def create_comparison_plots(results, output_dir, verbose=True):
    """
    Erstellt Vergleichs-Visualisierungen.

    Args:
        results: Dictionary mit Data-Objekten
        output_dir: Ausgabeverzeichnis
        verbose: Fortschritt ausgeben
    """

    data = results['original']

    if 'scan_id' not in data.df.columns or 'distance' not in data.df.columns:
        if verbose:
            print("  ⚠ Nicht genügend Daten für Visualisierungen")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot all scans (or cap to avoid too many lines)
    unique_ids = data.df['scan_id'].unique()
    max_plots = 10
    ids_to_plot = unique_ids[:max_plots]
    colors = plt.cm.tab10(np.linspace(0, 1, len(ids_to_plot)))

    for i, sid in enumerate(ids_to_plot):
        scan_data = data.df[data.df['scan_id'] == sid]
        if scan_data.empty:
            continue
        if 'angle' in scan_data.columns:
            axes[0, 0].plot(scan_data['angle'], scan_data['distance'], 'o-', markersize=2, alpha=0.6, color=colors[i], label=f'Scan {sid}')
            axes[0, 0].set_xlabel('Winkel (Grad)')
        else:
            axes[0, 0].plot(scan_data['distance'].values, 'o-', markersize=2, alpha=0.6, color=colors[i], label=f'Scan {sid}')
            axes[0, 0].set_xlabel('Messung')

    axes[0, 0].set_ylabel('Distanz')
    axes[0, 0].set_title('Original (multiple Scans)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Histogramm - Original (all scans)
    data.df['distance'].hist(bins=50, ax=axes[0, 1], edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Verteilung Distanzen (alle Scans)')
    axes[0, 1].set_xlabel('Distanz')
    axes[0, 1].set_ylabel('Häufigkeit')

    # Zentrierte Version (falls vorhanden) - plot multiple scans similarly
    if 'centered' in results and results['centered'] is not None:
        centered_data = results['centered']
        ids_centered = centered_data.df['scan_id'].unique() if 'scan_id' in centered_data.df.columns else []
        ids_centered_to_plot = ids_centered[:max_plots] if ids_centered.size else ids_to_plot
        for i, sid in enumerate(ids_centered_to_plot):
            scan_centered = centered_data.df[centered_data.df['scan_id'] == sid]
            if scan_centered.empty:
                continue
            if 'angle' in scan_centered.columns and 'normalized' in scan_centered.columns:
                axes[1, 0].plot(scan_centered['angle'], scan_centered['normalized'], 'o-', markersize=2, alpha=0.6, color=colors[i], label=f'Scan {sid}')
                axes[1, 0].set_xlabel('Winkel (Grad)')
            elif 'normalized' in scan_centered.columns:
                axes[1, 0].plot(scan_centered['normalized'].values, 'o-', markersize=2, alpha=0.6, color=colors[i], label=f'Scan {sid}')
                axes[1, 0].set_xlabel('Messung')

        axes[1, 0].set_ylabel('Normalisiert')
        axes[1, 0].set_title('Zentriert & Normalisiert (multiple Scans)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # MinMax (falls vorhanden)
    minmax_key = [k for k in results.keys() if 'minmax' in k]
    if minmax_key:
        minmax_data = results[minmax_key[0]]
        minmax_data.df['distance'].hist(bins=50, ax=axes[1, 1],
                                        edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_title(f'Verteilung nach {minmax_key[0]}')
        axes[1, 1].set_xlabel('Distanz (normalisiert)')
        axes[1, 1].set_ylabel('Häufigkeit')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'comparison_normalization.png', dpi=300, bbox_inches='tight')
    plt.close()


print("script starts...")
preprocessing_scan_data("/Users/maltehartmann/PycharmProjects/esp32-ml-object-recognition/training_pc/data/raw/objects/oval/oval.csv",
                        normalize_methods=["minmax", "standard", "robust"])
print("script finished")