import matplotlib.pyplot as plt
import numpy as np

def plot_speed_comparison():
    solvers = ['Traditional WLS', 'PFGNN']
    times_ms = [59.60, 22.06]
    colors = ['#2980b9', '#f39c12']
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(solvers, times_ms, color=colors, height=0.5, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Inference Latency (ms)', fontsize=11)
    ax.set_title('Computational Efficiency: IEEE 300-Bus State Estimation', fontsize=13, pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(0, 75)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f} ms', 
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    
    save_path = 'latency_comparison_300bus.png'
    plt.savefig(save_path, dpi=300, transparent=False)
    print(f"Comparison chart saved to: {save_path}")

if __name__ == "__main__":
    plot_speed_comparison()
