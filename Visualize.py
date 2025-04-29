import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(y_true, y_pred, threshold=0.5):
    
    # Flatten arrays for metric computation
    y_true_flat = y_true.flatten()
    y_pred_flat = (y_pred.flatten() > threshold).astype(np.int32)

    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    iou = jaccard_score(y_true_flat, y_pred_flat, zero_division=0)
    dice = 2 * np.sum(y_true_flat * y_pred_flat) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-8)
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    return acc, prec, rec, f1, iou, dice, cm


y_true_list = []
preds_list = []

for images, masks in test_ds:
    y_true_list.append(masks.numpy())
    preds_list.append(model.predict(images))

# Concatenate along the first dimension
y_test = np.concatenate(y_true_list, axis=0)
preds = np.concatenate(preds_list, axis=0)

# Compute metrics
acc, prec, rec, f1, iou, dice, cm = compute_metrics(y_test, preds, threshold=0.5)

print("Accuracy: {:.4f}".format(acc))
print("Precision: {:.4f}".format(prec))
print("Recall: {:.4f}".format(rec))
print("F1-Score: {:.4f}".format(f1))
print("IoU: {:.4f}".format(iou))
print("Dice Score: {:.4f}".format(dice))
print("Confusion Matrix:")
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()