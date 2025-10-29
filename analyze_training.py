"""
Analyze Training Logs and Visualize Overfitting Patterns
Usage: python analyze_training.py
"""

import re
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def parse_training_log(log_path):
    """Parse training log file and extract metrics"""
    
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract epoch metrics using regex
    # Pattern: Train Loss: X.XXXX, Val Loss: X.XXXX, Val Acc: X.XXXX, Val F1: X.XXXX
    pattern = r'Epoch (\d+)/\d+.*?Train Loss: ([\d.]+), Val Loss: ([\d.]+), Val Acc: ([\d.]+), Val F1: ([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        epoch, train_loss, val_loss, val_acc, val_f1 = match
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_accs.append(float(val_acc))
        val_f1s.append(float(val_f1))
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'val_f1': val_f1s
    }


def detect_overfitting(data):
    """Detect overfitting patterns in training data"""
    
    issues = []
    
    # Check 1: Validation loss increases after initial decrease
    if len(data['val_loss']) > 5:
        min_val_loss = min(data['val_loss'])
        min_epoch = data['val_loss'].index(min_val_loss) + 1
        final_val_loss = data['val_loss'][-1]
        
        if final_val_loss > min_val_loss * 1.5:
            issues.append(f"⚠️ Validation loss increased by {((final_val_loss/min_val_loss - 1) * 100):.1f}% after epoch {min_epoch}")
    
    # Check 2: Training loss very low while validation loss high
    if len(data['train_loss']) > 0:
        final_train_loss = data['train_loss'][-1]
        final_val_loss = data['val_loss'][-1]
        
        if final_train_loss < 0.1 and final_val_loss > 2.0:
            issues.append(f"⚠️ Large train/val gap: Train={final_train_loss:.4f}, Val={final_val_loss:.4f}")
    
    # Check 3: Validation accuracy plateaus
    if len(data['val_acc']) > 10:
        recent_accs = data['val_acc'][-10:]
        if max(recent_accs) - min(recent_accs) < 0.02:  # Less than 2% variation
            issues.append(f"⚠️ Validation accuracy plateaued at ~{sum(recent_accs)/len(recent_accs):.2%}")
    
    return issues


def plot_training_metrics(data, output_path='training_analysis.png'):
    """Create comprehensive training visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Analysis - Overfitting Detection', fontsize=16, fontweight='bold')
    
    epochs = data['epochs']
    
    # Plot 1: Loss Comparison
    ax1 = axes[0, 0]
    ax1.plot(epochs, data['train_loss'], label='Training Loss', marker='o', linewidth=2)
    ax1.plot(epochs, data['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves - Overfitting Indicator', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Mark best validation loss
    min_val_idx = data['val_loss'].index(min(data['val_loss']))
    ax1.axvline(x=epochs[min_val_idx], color='red', linestyle='--', 
                label=f'Best Val Loss (Epoch {epochs[min_val_idx]})', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Plot 2: Validation Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, [acc * 100 for acc in data['val_acc']], 
             marker='o', linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark best accuracy
    max_acc_idx = data['val_acc'].index(max(data['val_acc']))
    ax2.axvline(x=epochs[max_acc_idx], color='red', linestyle='--', 
                label=f'Best Acc (Epoch {epochs[max_acc_idx]})', alpha=0.7)
    ax2.legend(fontsize=10)
    
    # Plot 3: Train/Val Loss Gap (Overfitting Metric)
    ax3 = axes[1, 0]
    loss_gap = [val - train for train, val in zip(data['train_loss'], data['val_loss'])]
    ax3.plot(epochs, loss_gap, marker='o', linewidth=2, color='red')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss Gap (Val - Train)', fontsize=12)
    ax3.set_title('Overfitting Metric (Higher = More Overfitting)', fontsize=14, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(epochs, loss_gap, alpha=0.3, color='red')
    
    # Plot 4: F1 Score
    ax4 = axes[1, 1]
    ax4.plot(epochs, [f1 * 100 for f1 in data['val_f1']], 
             marker='o', linewidth=2, color='purple')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation F1 Score (%)', fontsize=12)
    ax4.set_title('Validation F1 Score Over Time', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    plt.show()


def generate_report(data):
    """Generate text report with recommendations"""
    
    print("\n" + "="*80)
    print("📊 TRAINING ANALYSIS REPORT")
    print("="*80)
    
    # Basic statistics
    print(f"\n📈 Training Statistics:")
    print(f"  Total Epochs: {len(data['epochs'])}")
    print(f"  Initial Train Loss: {data['train_loss'][0]:.4f}")
    print(f"  Final Train Loss: {data['train_loss'][-1]:.4f}")
    print(f"  Loss Reduction: {((1 - data['train_loss'][-1]/data['train_loss'][0]) * 100):.1f}%")
    
    print(f"\n📉 Validation Performance:")
    print(f"  Best Val Loss: {min(data['val_loss']):.4f} (Epoch {data['val_loss'].index(min(data['val_loss'])) + 1})")
    print(f"  Final Val Loss: {data['val_loss'][-1]:.4f}")
    print(f"  Best Val Accuracy: {max(data['val_acc']):.4f} ({max(data['val_acc'])*100:.2f}%)")
    print(f"  Final Val Accuracy: {data['val_acc'][-1]:.4f} ({data['val_acc'][-1]*100:.2f}%)")
    print(f"  Best Val F1: {max(data['val_f1']):.4f}")
    
    # Overfitting analysis
    print(f"\n⚠️  Overfitting Analysis:")
    issues = detect_overfitting(data)
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ No major overfitting patterns detected!")
    
    # Train/Val gap
    final_gap = data['val_loss'][-1] - data['train_loss'][-1]
    print(f"\n📊 Train/Val Loss Gap:")
    print(f"  Final Gap: {final_gap:.4f}")
    if final_gap > 2.0:
        print(f"  Status: 🔴 SEVERE OVERFITTING")
    elif final_gap > 1.0:
        print(f"  Status: 🟡 MODERATE OVERFITTING")
    else:
        print(f"  Status: 🟢 GOOD GENERALIZATION")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if final_gap > 2.0:
        print("  1. ✅ Increase dropout rates (Applied)")
        print("  2. ✅ Reduce early stopping patience (Applied)")
        print("  3. 🔄 Consider freezing BERT layers")
        print("  4. 🔄 Reduce model complexity (smaller LSTM/CNN)")
        print("  5. 🔄 Add data augmentation")
        print("  6. 🔄 Increase weight decay further")
    
    print("\n" + "="*80)


def main():
    """Main analysis function"""
    
    # Find training log
    log_path = Path('experiments/logs/training.log')
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return
    
    print(f"📂 Reading log file: {log_path}")
    data = parse_training_log(log_path)
    
    if not data['epochs']:
        print("❌ No training data found in log file")
        return
    
    print(f"✅ Found data for {len(data['epochs'])} epochs")
    
    # Generate visualizations
    plot_training_metrics(data)
    
    # Generate report
    generate_report(data)
    
    # Create DataFrame for easy viewing
    df = pd.DataFrame(data)
    print("\n📋 Training Data Summary:")
    print(df.describe())
    
    # Save to CSV
    csv_path = 'experiments/logs/training_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Data saved to: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
