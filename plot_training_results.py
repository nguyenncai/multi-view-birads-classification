"""
Plot training results from CSV file
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def plot_training_results(csv_path, output_dir=None):
    """Plot training metrics from CSV file"""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Task-specific losses
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['train_diagnosis'], label='Train Diagnosis', marker='o', markersize=3)
    ax.plot(df['epoch'], df['val_diagnosis'], label='Val Diagnosis', marker='s', markersize=3)
    ax.plot(df['epoch'], df['train_birads'], label='Train BI-RADS', marker='^', markersize=3)
    ax.plot(df['epoch'], df['val_birads'], label='Val BI-RADS', marker='v', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Task-Specific Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: F1 Diagnosis (Right vs Left)
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['val_f1_diagnosis_right'], label='Right', marker='o', markersize=3)
    ax.plot(df['epoch'], df['val_f1_diagnosis_left'], label='Left', marker='s', markersize=3)
    ax.plot(df['epoch'], df['val_f1_diagnosis_avg'], label='Average', marker='^', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score - Diagnosis Task')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 4: F1 BI-RADS (Right vs Left)
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['val_f1_birads_right'], label='Right', marker='o', markersize=3)
    ax.plot(df['epoch'], df['val_f1_birads_left'], label='Left', marker='s', markersize=3)
    ax.plot(df['epoch'], df['val_f1_birads_avg'], label='Average', marker='^', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score - BI-RADS Task')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 5: Overall F1 Macro
    ax = axes[2, 0]
    ax.plot(df['epoch'], df['val_f1_macro_overall'], label='F1 Macro Overall', 
            marker='o', markersize=4, linewidth=2, color='green')
    ax.axhline(y=df['val_f1_macro_overall'].max(), color='r', linestyle='--', 
               label=f'Best: {df["val_f1_macro_overall"].max():.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Overall F1 Macro (Average of Both Tasks)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 6: Attention Weights
    ax = axes[2, 1]
    ax.plot(df['epoch'], df['attention_lcc'], label='L_CC', marker='o', markersize=3)
    ax.plot(df['epoch'], df['attention_lmlo'], label='L_MLO', marker='s', markersize=3)
    ax.plot(df['epoch'], df['attention_rcc'], label='R_CC', marker='^', markersize=3)
    ax.plot(df['epoch'], df['attention_rmlo'], label='R_MLO', marker='v', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention Weights per View')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    best_f1_idx = df['val_f1_macro_overall'].idxmax()
    best_loss_idx = df['val_loss'].idxmin()
    
    print(f"\nBest F1 Macro Overall: {df.loc[best_f1_idx, 'val_f1_macro_overall']:.4f} (Epoch {df.loc[best_f1_idx, 'epoch']})")
    print(f"  - F1 Diagnosis (R/L/Avg): {df.loc[best_f1_idx, 'val_f1_diagnosis_right']:.4f}/{df.loc[best_f1_idx, 'val_f1_diagnosis_left']:.4f}/{df.loc[best_f1_idx, 'val_f1_diagnosis_avg']:.4f}")
    print(f"  - F1 BI-RADS (R/L/Avg): {df.loc[best_f1_idx, 'val_f1_birads_right']:.4f}/{df.loc[best_f1_idx, 'val_f1_birads_left']:.4f}/{df.loc[best_f1_idx, 'val_f1_birads_avg']:.4f}")
    print(f"  - Val Loss: {df.loc[best_f1_idx, 'val_loss']:.4f}")
    
    print(f"\nBest Validation Loss: {df.loc[best_loss_idx, 'val_loss']:.4f} (Epoch {df.loc[best_loss_idx, 'epoch']})")
    print(f"  - F1 Macro Overall: {df.loc[best_loss_idx, 'val_f1_macro_overall']:.4f}")
    
    print(f"\nFinal Epoch ({df['epoch'].iloc[-1]}):")
    print(f"  - Val Loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"  - F1 Macro Overall: {df['val_f1_macro_overall'].iloc[-1]:.4f}")
    
    print(f"\nAverage Attention Weights (Last Epoch):")
    print(f"  - L_CC: {df['attention_lcc'].iloc[-1]:.4f}")
    print(f"  - L_MLO: {df['attention_lmlo'].iloc[-1]:.4f}")
    print(f"  - R_CC: {df['attention_rcc'].iloc[-1]:.4f}")
    print(f"  - R_MLO: {df['attention_rmlo'].iloc[-1]:.4f}")
    
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to training_results.csv')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plot_training_results(args.csv_path, args.output_dir)
