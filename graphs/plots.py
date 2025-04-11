import matplotlib.pyplot as plt

# Data for Split 3 (accuracy over 10 epochs)
epochs = list(range(1, 11))
train_acc_split3 = [0.0929, 0.6286, 0.8929, 0.9357, 0.9643, 0.9429, 0.9357, 0.9357, 0.9500, 0.9571]
test_acc_split3 = [0.5000, 0.7143, 0.7857, 0.8214, 0.8214, 0.7857, 0.7857, 0.7857, 0.7857, 0.7857]

# Line plot for Split 3 accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc_split3, label='Train Accuracy', marker='o')
plt.plot(epochs, test_acc_split3, label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Split 3: Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/split3_accuracy_over_epochs.png')
plt.show()

# Data for Split 1 and 2 (final results)
splits = ['Split 1', 'Split 2']
train_accuracies = [0.8860, 0.9316]
test_accuracies = [0.5926, 0.5882]

# Bar chart comparing accuracy
x = range(len(splits))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - width/2 for i in x], train_accuracies, width, label='Train Accuracy')
for i, acc in enumerate(train_accuracies):
    plt.text(i - width/2, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
plt.bar([i + width/2 for i in x], test_accuracies, width, label='Test Accuracy')
for i, acc in enumerate(test_accuracies):
    plt.text(i + width/2, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
plt.xticks(x, splits)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy: Split 1 vs Split 2')
plt.legend()
plt.tight_layout()
plt.savefig('./graphs/split1_vs_split2_comparison.png')
plt.show()