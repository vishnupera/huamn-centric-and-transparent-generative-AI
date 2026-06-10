import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "research_study/results"

def analyze():
    csv_path = f"{OUTPUT_DIR}/final_results.csv"
    if not os.path.exists(csv_path):
        print("No results found. Run experiment first.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Comparison Tables
    print("\n=== Overall Comparison Table ===")
    print(df.groupby('model')[['bleu', 'rougeL', 'faithfulness', 'hallucination_rate', 'avg_transparency']].mean())
    
    # Save per-dataset tables
    for dataset in df['dataset'].unique():
        sub_df = df[df['dataset'] == dataset]
        sub_df.to_csv(f"{OUTPUT_DIR}/table_{dataset}.csv", index=False)
    
    # 2. Plots
    sns.set_theme(style="whitegrid")
    
    # Metric Comparisons
    metrics_to_plot = ['bleu', 'rougeL', 'faithfulness', 'hallucination_rate', 'avg_latency', 'avg_transparency']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='dataset', y=metric, hue='model')
        plt.title(f"{metric.upper()} Comparison by Dataset")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plot_{metric}.png")
        plt.close()

    # Faithfulness vs Hallucination Scatter (if applicable, though they are derived)
    # Let's do a Efficiency plot: Latency vs BLEU
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='avg_latency', y='bleu', hue='model', style='dataset', s=100)
    plt.title("Efficiency Frontier: Latency vs Quality")
    plt.savefig(f"{OUTPUT_DIR}/plot_efficiency.png")
    plt.close()

    print("Plots generated in results directory.")

if __name__ == "__main__":
    analyze()
