"""
Generate a detailed methodology diagram for the CV Screening System
This illustrates the complete workflow from data input to prediction output
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines
import numpy as np

# Set up the figure with high quality
plt.figure(figsize=(20, 28))
ax = plt.gca()
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')

# Color scheme
color_data = '#E3F2FD'  # Light blue - Data processing
color_model = '#FFF3E0'  # Light orange - Model components
color_training = '#F3E5F5'  # Light purple - Training
color_inference = '#E8F5E9'  # Light green - Inference
color_output = '#FFF9C4'  # Light yellow - Output
color_arrow = '#424242'  # Dark gray - Arrows

# Title
plt.text(10, 27.5, 'CV SCREENING SYSTEM - METHODOLOGY DIAGRAM', 
         ha='center', va='top', fontsize=24, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#1E3A8A', edgecolor='black', linewidth=2, alpha=0.9),
         color='white')

plt.text(10, 26.7, 'Hybrid Deep Learning Approach for Resume-Job Matching', 
         ha='center', va='top', fontsize=14, style='italic', color='#1E3A8A')

# ============================================================================
# SECTION 1: DATA INPUT & PREPROCESSING (Top)
# ============================================================================
y_pos = 25.5

# Section header
plt.text(10, y_pos, 'PHASE 1: DATA ACQUISITION & PREPROCESSING', 
         ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#10B981', edgecolor='black', linewidth=2),
         color='white')

y_pos -= 1.2

# Raw Data Input
box1 = FancyBboxPatch((1, y_pos-0.8), 4, 1.6, boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(box1)
plt.text(3, y_pos, 'RAW DATA INPUT', ha='center', va='center', 
         fontsize=12, fontweight='bold')
plt.text(3, y_pos-0.35, '• Resume.csv Dataset\n• 2,483 CV samples\n• Multiple job categories', 
         ha='center', va='center', fontsize=9)

# Data Cleaning
box2 = FancyBboxPatch((6, y_pos-0.8), 4, 1.6, boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(box2)
plt.text(8, y_pos, 'DATA CLEANING', ha='center', va='center', 
         fontsize=12, fontweight='bold')
plt.text(8, y_pos-0.35, '• Remove URLs, emails\n• Lowercase conversion\n• Special char handling', 
         ha='center', va='center', fontsize=9)

# Feature Extraction
box3 = FancyBboxPatch((11, y_pos-0.8), 4, 1.6, boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(box3)
plt.text(13, y_pos, 'FEATURE EXTRACTION', ha='center', va='center', 
         fontsize=12, fontweight='bold')
plt.text(13, y_pos-0.35, '• Skill extraction\n• Experience parsing\n• Text statistics', 
         ha='center', va='center', fontsize=9)

# Data Split
box4 = FancyBboxPatch((16, y_pos-0.8), 3, 1.6, boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(box4)
plt.text(17.5, y_pos, 'DATA SPLIT', ha='center', va='center', 
         fontsize=12, fontweight='bold')
plt.text(17.5, y_pos-0.35, 'Train: 70%\nVal: 15%\nTest: 15%', 
         ha='center', va='center', fontsize=9)

# Arrows for data flow
arrow1 = FancyArrowPatch((5, y_pos), (6, y_pos), arrowstyle='->', mutation_scale=30, 
                        linewidth=2.5, color=color_arrow)
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((10, y_pos), (11, y_pos), arrowstyle='->', mutation_scale=30, 
                        linewidth=2.5, color=color_arrow)
ax.add_patch(arrow2)
arrow3 = FancyArrowPatch((15, y_pos), (16, y_pos), arrowstyle='->', mutation_scale=30, 
                        linewidth=2.5, color=color_arrow)
ax.add_patch(arrow3)

# ============================================================================
# SECTION 2: HYBRID MODEL ARCHITECTURE
# ============================================================================
y_pos -= 2.5

# Section header
plt.text(10, y_pos, 'PHASE 2: HYBRID MODEL ARCHITECTURE', 
         ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F59E0B', edgecolor='black', linewidth=2),
         color='white')

y_pos -= 1.2

# Input Layer
input_box = FancyBboxPatch((8, y_pos-0.5), 4, 1, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#BBDEFB', linewidth=2)
ax.add_patch(input_box)
plt.text(10, y_pos, 'INPUT TEXT\n(Resume Content)', ha='center', va='center', 
         fontsize=11, fontweight='bold')

y_pos -= 1.8

# Four parallel components
# BERT Component
bert_box = FancyBboxPatch((1, y_pos-1.5), 3.5, 3, boxstyle="round,pad=0.15",
                          edgecolor='#1565C0', facecolor=color_model, linewidth=3)
ax.add_patch(bert_box)
plt.text(2.75, y_pos+0.9, 'BERT COMPONENT', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='#1565C0')
plt.text(2.75, y_pos+0.3, 'DistilBERT\n(Semantic)\n\n• 768 hidden units\n• 2 layers\n• Dropout: 0.4\n\nWeight: 40%', 
         ha='center', va='center', fontsize=8)

# CNN Component
cnn_box = FancyBboxPatch((5.5, y_pos-1.5), 3.5, 3, boxstyle="round,pad=0.15",
                         edgecolor='#C62828', facecolor=color_model, linewidth=3)
ax.add_patch(cnn_box)
plt.text(7.25, y_pos+0.9, 'CNN COMPONENT', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='#C62828')
plt.text(7.25, y_pos+0.3, 'Convolutional\n(Local Patterns)\n\n• Filters: [3,4,5]\n• 100 each\n• Dropout: 0.5\n\nWeight: 25%', 
         ha='center', va='center', fontsize=8)

# LSTM Component
lstm_box = FancyBboxPatch((10, y_pos-1.5), 3.5, 3, boxstyle="round,pad=0.15",
                          edgecolor='#6A1B9A', facecolor=color_model, linewidth=3)
ax.add_patch(lstm_box)
plt.text(11.75, y_pos+0.9, 'LSTM COMPONENT', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='#6A1B9A')
plt.text(11.75, y_pos+0.3, 'Bi-LSTM\n(Sequential)\n\n• 256 hidden units\n• 2 layers\n• Attention\n\nWeight: 25%', 
         ha='center', va='center', fontsize=8)

# Traditional ML Component
trad_box = FancyBboxPatch((14.5, y_pos-1.5), 4, 3, boxstyle="round,pad=0.15",
                          edgecolor='#2E7D32', facecolor=color_model, linewidth=3)
ax.add_patch(trad_box)
plt.text(16.5, y_pos+0.9, 'TRADITIONAL ML', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='#2E7D32')
plt.text(16.5, y_pos+0.3, 'TF-IDF + Features\n(Statistical)\n\n• 10K features\n• N-grams (1-3)\n• Skills/Experience\n\nWeight: 10%', 
         ha='center', va='center', fontsize=8)

# Arrows from input to components
for x_target in [2.75, 7.25, 11.75, 16.5]:
    arrow = FancyArrowPatch((10, y_pos+1.3), (x_target, y_pos+1.5), 
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=2, color=color_arrow)
    ax.add_patch(arrow)

y_pos -= 3.5

# Attention & Fusion Layer
fusion_box = FancyBboxPatch((6, y_pos-0.8), 8, 1.6, boxstyle="round,pad=0.15",
                            edgecolor='black', facecolor='#FFE0B2', linewidth=3)
ax.add_patch(fusion_box)
plt.text(10, y_pos-0.15, 'ATTENTION & FUSION LAYER', ha='center', va='center', 
         fontsize=12, fontweight='bold')
plt.text(10, y_pos-0.55, 'Learnable weighted combination with self-attention (128-dim)', 
         ha='center', va='center', fontsize=9, style='italic')

# Arrows from components to fusion
for x_source in [2.75, 7.25, 11.75, 16.5]:
    arrow = FancyArrowPatch((x_source, y_pos+0.8), (10, y_pos-0.8), 
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=2, color=color_arrow, linestyle='--')
    ax.add_patch(arrow)

y_pos -= 1.5

# Output Layer
output_box = FancyBboxPatch((7, y_pos-0.6), 6, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#C8E6C9', linewidth=2)
ax.add_patch(output_box)
plt.text(10, y_pos, 'OUTPUT LAYER\n(Softmax Classification → Job Category)', 
         ha='center', va='center', fontsize=11, fontweight='bold')

# Arrow to output
arrow_out = FancyArrowPatch((10, y_pos-1.1), (10, y_pos-0.6), 
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=2.5, color=color_arrow)
ax.add_patch(arrow_out)

# ============================================================================
# SECTION 3: TRAINING PROCESS
# ============================================================================
y_pos -= 2

# Section header
plt.text(10, y_pos, 'PHASE 3: TRAINING PIPELINE', 
         ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#9C27B0', edgecolor='black', linewidth=2),
         color='white')

y_pos -= 1.2

# Training Configuration
train_config = FancyBboxPatch((1, y_pos-1.2), 5.5, 2.4, boxstyle="round,pad=0.15",
                              edgecolor='black', facecolor=color_training, linewidth=2)
ax.add_patch(train_config)
plt.text(3.75, y_pos+0.6, 'TRAINING CONFIG', ha='center', va='center', 
         fontsize=11, fontweight='bold')
plt.text(3.75, y_pos-0.2, '• Batch Size: 8\n• Learning Rate: 2e-5\n• Weight Decay: 0.05\n• Max Epochs: 20\n• Mixed Precision: FP16\n• GPU: CUDA Accelerated', 
         ha='center', va='center', fontsize=8)

# Optimization Strategy
opt_box = FancyBboxPatch((7.5, y_pos-1.2), 5, 2.4, boxstyle="round,pad=0.15",
                         edgecolor='black', facecolor=color_training, linewidth=2)
ax.add_patch(opt_box)
plt.text(10, y_pos+0.6, 'OPTIMIZATION', ha='center', va='center', 
         fontsize=11, fontweight='bold')
plt.text(10, y_pos-0.2, '• AdamW Optimizer\n• Linear warmup (500 steps)\n• Gradient clipping (1.0)\n• Early stopping (patience=3)\n• L2 regularization', 
         ha='center', va='center', fontsize=8)

# Regularization
reg_box = FancyBboxPatch((13.5, y_pos-1.2), 5.5, 2.4, boxstyle="round,pad=0.15",
                         edgecolor='black', facecolor=color_training, linewidth=2)
ax.add_patch(reg_box)
plt.text(16.25, y_pos+0.6, 'REGULARIZATION', ha='center', va='center', 
         fontsize=11, fontweight='bold')
plt.text(16.25, y_pos-0.2, '• Dropout (0.3-0.5)\n• Weight decay (0.05)\n• Data augmentation\n• Cross-validation ready\n• Prevents overfitting', 
         ha='center', va='center', fontsize=8)

y_pos -= 3

# Training Monitoring
monitor_box = FancyBboxPatch((3, y_pos-0.8), 14, 1.6, boxstyle="round,pad=0.15",
                             edgecolor='black', facecolor='#E1BEE7', linewidth=2)
ax.add_patch(monitor_box)
plt.text(10, y_pos-0.15, 'TRAINING MONITORING & EXPERIMENT TRACKING', ha='center', va='center', 
         fontsize=12, fontweight='bold')
plt.text(10, y_pos-0.55, 'Logs: Loss, Accuracy, Precision, Recall, F1-Score | Checkpoints | Learning curves | Confusion matrices', 
         ha='center', va='center', fontsize=9)

# ============================================================================
# SECTION 4: EVALUATION & VALIDATION
# ============================================================================
y_pos -= 2

# Section header
plt.text(10, y_pos, 'PHASE 4: EVALUATION & VALIDATION', 
         ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#0288D1', edgecolor='black', linewidth=2),
         color='white')

y_pos -= 1.2

# Performance Metrics
metrics_box = FancyBboxPatch((1.5, y_pos-1.2), 8, 2.4, boxstyle="round,pad=0.15",
                             edgecolor='black', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(metrics_box)
plt.text(5.5, y_pos+0.7, 'PERFORMANCE METRICS', ha='center', va='center', 
         fontsize=11, fontweight='bold')
plt.text(5.5, y_pos-0.1, 'Test Accuracy: 85.25%\nPrecision: 85.46%\nRecall: 85.25%\nF1-Score: 84.56%\n\nNo Overfitting Detected!', 
         ha='center', va='center', fontsize=9)

# Visualization Outputs
viz_box = FancyBboxPatch((10.5, y_pos-1.2), 8, 2.4, boxstyle="round,pad=0.15",
                         edgecolor='black', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(viz_box)
plt.text(14.5, y_pos+0.7, 'VISUALIZATIONS', ha='center', va='center', 
         fontsize=11, fontweight='bold')
plt.text(14.5, y_pos-0.1, '• Training/Validation curves\n• Confusion matrix\n• Per-class performance\n• Component weight analysis\n• Convergence plots', 
         ha='center', va='center', fontsize=9)

# ============================================================================
# SECTION 5: INFERENCE & DEPLOYMENT
# ============================================================================
y_pos -= 3

# Section header
plt.text(10, y_pos, 'PHASE 5: INFERENCE & DEPLOYMENT', 
         ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#388E3C', edgecolor='black', linewidth=2),
         color='white')

y_pos -= 1.2

# FastAPI Backend
backend_box = FancyBboxPatch((1, y_pos-1), 5.5, 2, boxstyle="round,pad=0.15",
                             edgecolor='black', facecolor=color_inference, linewidth=2)
ax.add_patch(backend_box)
plt.text(3.75, y_pos+0.4, 'FASTAPI BACKEND', ha='center', va='center', 
         fontsize=11, fontweight='bold')
plt.text(3.75, y_pos-0.3, '• RESTful API endpoints\n• Text extraction (PDF/DOCX)\n• Model inference\n• Result ranking\n• CORS enabled', 
         ha='center', va='center', fontsize=8)

# Model Prediction
pred_box = FancyBboxPatch((7.5, y_pos-1), 5, 2, boxstyle="round,pad=0.15",
                          edgecolor='black', facecolor=color_inference, linewidth=2)
ax.add_patch(pred_box)
plt.text(10, y_pos+0.4, 'MODEL PREDICTION', ha='center', va='center', 
         fontsize=11, fontweight='bold')
plt.text(10, y_pos-0.3, '• Load trained model\n• Process input text\n• Generate embeddings\n• Classify job category\n• Confidence scores', 
         ha='center', va='center', fontsize=8)

# Next.js Frontend
frontend_box = FancyBboxPatch((13.5, y_pos-1), 5.5, 2, boxstyle="round,pad=0.15",
                              edgecolor='black', facecolor=color_inference, linewidth=2)
ax.add_patch(frontend_box)
plt.text(16.25, y_pos+0.4, 'NEXT.JS FRONTEND', ha='center', va='center', 
         fontsize=11, fontweight='bold')
plt.text(16.25, y_pos-0.3, '• File upload interface\n• Job description input\n• Real-time matching\n• Results visualization\n• Responsive UI', 
         ha='center', va='center', fontsize=8)

# Arrows showing flow
arrow_deploy1 = FancyArrowPatch((6.5, y_pos), (7.5, y_pos), arrowstyle='<->', mutation_scale=25, 
                               linewidth=2, color=color_arrow)
ax.add_patch(arrow_deploy1)
arrow_deploy2 = FancyArrowPatch((12.5, y_pos), (13.5, y_pos), arrowstyle='<->', mutation_scale=25, 
                               linewidth=2, color=color_arrow)
ax.add_patch(arrow_deploy2)

# ============================================================================
# SECTION 6: OUTPUT & RESULTS
# ============================================================================
y_pos -= 2.5

# Section header
plt.text(10, y_pos, 'PHASE 6: OUTPUT & RESULTS', 
         ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F57C00', edgecolor='black', linewidth=2),
         color='white')

y_pos -= 1.2

# Final Output
output_final = FancyBboxPatch((3, y_pos-1.2), 14, 2.4, boxstyle="round,pad=0.15",
                              edgecolor='black', facecolor=color_output, linewidth=3)
ax.add_patch(output_final)
plt.text(10, y_pos+0.6, 'FINAL OUTPUT', ha='center', va='center', 
         fontsize=12, fontweight='bold')
plt.text(10, y_pos-0.2, '✓ Ranked list of candidates matched to job requirements\n✓ Match scores with confidence levels\n✓ Skill alignment analysis\n✓ Top candidate recommendations\n✓ Detailed matching report', 
         ha='center', va='center', fontsize=9)

# ============================================================================
# KEY FEATURES & ADVANTAGES (Side panel)
# ============================================================================

# Features box
features_box = FancyBboxPatch((0.3, 0.5), 6, 4.5, boxstyle="round,pad=0.15",
                              edgecolor='#1E3A8A', facecolor='#E8EAF6', linewidth=2)
ax.add_patch(features_box)
plt.text(3.3, 4.6, '🔑 KEY FEATURES', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='#1E3A8A')

features_text = """
✓ Hybrid deep learning architecture
✓ Multi-component fusion approach
✓ GPU-accelerated training (CUDA)
✓ Mixed precision (FP16) support
✓ Advanced regularization techniques
✓ Real-time inference capability
✓ Scalable REST API
✓ Modern web interface
✓ Document parsing (PDF/DOCX)
✓ Comprehensive evaluation metrics
"""
plt.text(3.3, 2.5, features_text, ha='center', va='center', 
         fontsize=8, family='monospace')

# Technical Specs box
specs_box = FancyBboxPatch((7, 0.5), 6, 4.5, boxstyle="round,pad=0.15",
                           edgecolor='#1E3A8A', facecolor='#E8EAF6', linewidth=2)
ax.add_patch(specs_box)
plt.text(10, 4.6, '⚙️ TECHNICAL SPECS', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='#1E3A8A')

specs_text = """
Framework: PyTorch 2.0+
NLP Model: DistilBERT
Backend: FastAPI
Frontend: Next.js 14 + TypeScript
Database: Processed features (NumPy)
GPU: NVIDIA CUDA 12.1
Precision: Mixed FP16/FP32
Dataset: 2,483 CV samples
Categories: Multiple job roles
Training Time: ~2-3 hours (RTX 3050)
"""
plt.text(10, 2.5, specs_text, ha='center', va='center', 
         fontsize=8, family='monospace')

# Performance box
perf_box = FancyBboxPatch((13.7, 0.5), 6, 4.5, boxstyle="round,pad=0.15",
                          edgecolor='#1E3A8A', facecolor='#E8EAF6', linewidth=2)
ax.add_patch(perf_box)
plt.text(16.7, 4.6, '📊 PERFORMANCE', ha='center', va='center', 
         fontsize=11, fontweight='bold', color='#1E3A8A')

perf_text = """
✓ Test Accuracy: 85.25%
✓ Precision: 85.46%
✓ Recall: 85.25%
✓ F1-Score: 84.56%

Generalization:
• Train-Val gap: < 1%
• No overfitting detected
• Stable convergence
• Robust across classes

Inference: < 500ms per CV
"""
plt.text(16.7, 2.5, perf_text, ha='center', va='center', 
         fontsize=8, family='monospace')

# ============================================================================
# Legend and Additional Info
# ============================================================================

# Add workflow direction indicator
plt.text(10, 0.2, '↓ Data flows from top to bottom through the complete pipeline ↓', 
         ha='center', va='center', fontsize=10, style='italic', color='#424242')

# Add timestamp and version
plt.text(1, 0.05, 'Generated: October 2025 | Version: 1.0', 
         ha='left', va='bottom', fontsize=8, color='gray')
plt.text(19, 0.05, 'Project: cv-screening-tool', 
         ha='right', va='bottom', fontsize=8, color='gray')

# Save the diagram
plt.tight_layout()
plt.savefig('methodology_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('methodology_diagram.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("\n" + "="*80)
print("METHODOLOGY DIAGRAM GENERATED SUCCESSFULLY")
print("="*80)
print(f"\n✓ High-resolution PNG saved: methodology_diagram.png")
print(f"✓ Vector PDF saved: methodology_diagram.pdf")
print(f"\nThe diagram illustrates:")
print("  • Complete data processing pipeline")
print("  • Hybrid model architecture with 4 components")
print("  • Training and optimization strategies")
print("  • Evaluation and validation procedures")
print("  • Deployment and inference workflow")
print("  • Key features and performance metrics")
print("\n" + "="*80)
