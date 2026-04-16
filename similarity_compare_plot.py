import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_path="tune_selector_results.csv"):
    df = pd.read_csv(csv_path)

    df_sorted = df.sort_values(by="test_acc", ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_sorted["config"], df_sorted["test_acc"], color='skyblue', edgecolor='navy')

    plt.title("Similarity Measures Ranked by Test Accuracy (30 epochs)", fontsize=14)
    plt.ylabel("Final Test Accuracy", fontsize=12)
    plt.ylim(0, 1.1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    
    # Save the file
    plt.savefig("similarity_results_plot.png", dpi=300)

if __name__ == "__main__":
    plot_results()