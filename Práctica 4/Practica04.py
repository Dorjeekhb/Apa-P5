from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from utils import load_data, one_hot_encoding
from public_test import checkNNGradients, MLP_test_step
from sklearn.model_selection import train_test_split
import numpy as np
import os

"""
Test 1 to be executed in Main
"""
def gradientTest():
    checkNNGradients(costNN, target_gradient, 0)
    checkNNGradients(costNN, target_gradient, 1)

"""
Test 2 to be executed in Main
"""
def MLP_test(X_train, y_train, X_test, y_test, iters, verbose):
    print("We assume that: random_state of train_test_split = 0 alpha=1, num_iterations = 2000, test_size=0.33, seed=0 and epislom = 0.12")
    print("Test 1 Calculando para lambda = 0")
    MLP_test_step(MLP_backprop_predict, 1, X_train, y_train, X_test, y_test, 0,   iters, 0.92606, verbose)
    print("Test 2 Calculando para lambda = 0.5")
    MLP_test_step(MLP_backprop_predict, 1, X_train, y_train, X_test, y_test, 0.5, iters, 0.92545, verbose)
    print("Test 3 Calculando para lambda = 1")
    MLP_test_step(MLP_backprop_predict, 1, X_train, y_train, X_test, y_test, 1,   iters, 0.92667, verbose)

def main():
    print("Main program")

    # Modo FULL (2000 iter) o FAST (50 iter) con variable de entorno APA_FAST=1
    fast = os.getenv("APA_FAST", "0") == "1"
    iters = 50 if fast else 2000
    verbose = 10 if fast else 200
    print(f"[Config] Using {'FAST' if fast else 'FULL'} mode: iterations = {iters}")

    # Test 1: verificación de gradientes
    gradientTest()

    # Carga de datos (ruta relativa al archivo)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    X, y = load_data(os.path.join(base_dir, "data", "ex3data1.mat"))

    # One-hot para entrenamiento
    Y = one_hot_encoding(y)

    # Split train/test
    X_train, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.33, random_state=0, stratify=y)
    Y_train = one_hot_encoding(y_train_raw)

    y_test = np.where(y_test == 10, 0, y_test)

    # Test 2: entrenamiento y accuracies esperadas
    MLP_test(X_train, Y_train, X_test, y_test, iters, verbose)

if __name__ == "__main__":
    main()
