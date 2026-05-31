import matplotlib.pyplot as plt

# Model names and their corresponding AUC scores
models = ['Supervised Baseline\n(MobileNetV2)', 'Metric Learning\n(Triplet ResNet18)']
auc_scores = [0.7516, 0.8883]

plt.figure(figsize=(8, 5))

# Create bars
bars = plt.bar(models, auc_scores, width=0.5, edgecolor='black', alpha=0.85)

# Add titles and labels
plt.title('Face Verification Performance Comparison (ROC-AUC)', fontsize=14, pad=15)
plt.ylabel('Area Under the ROC Curve (AUC)', fontsize=12)
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.4f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Save the visualization
plt.savefig('reports/Patrick_Model_Comparison_AUC.png', bbox_inches='tight', dpi=300)
print("Comparison chart successfully saved as 'Patrick_Model_Comparison_AUC.png'")