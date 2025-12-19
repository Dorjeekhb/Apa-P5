import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_weights,one_hot_encoding, accuracy
from MLP import MLP
from public_test import compute_cost_test, predict_test
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


x,y = load_data('data/ex3data1.mat')
theta1, theta2 = load_weights('data/ex3weights.mat')

mlp = MLP(theta1, theta2)

a1, a2, a3, z2, z3 = mlp.feedforward(x)
predictions = mlp.predict(a3)

acc = accuracy(predictions, y)
print(f"Accuracy: {acc*100:.2f}%")

y_encoded = one_hot_encoding(y)
cost = mlp.compute_cost(a3, y_encoded)
print(f"Cost: {cost}")

predict_test(predictions, y, accuracy)
compute_cost_test(mlp, a3, y_encoded)

y_binary = (y == 0).astype(int)
predictions_binary = (predictions == 0).astype(int)

cm = confusion_matrix(y_binary, predictions_binary)
print("\nConfusion Matrix for digit 0:")
print(cm)

precision = precision_score(y_binary, predictions_binary)
recall = recall_score(y_binary, predictions_binary)
f1 = f1_score(y_binary, predictions_binary)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")