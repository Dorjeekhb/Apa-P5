import numpy as np
import copy
import math

from LinearRegressionMulti import LinearRegMulti

class LogisticRegMulti(LinearRegMulti):

    """
    Regresión Logística Multivariable.
    Hereda de LinearRegMulti y sobrescribe los métodos necesarios para 
    implementar regresión logística con regularización L2.

    Args:
        x (ndarray): Shape (m,n) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction (0 o 1)
        w (ndarray): Shape (n,) Parameters of the model
        b (scalar): Parameter of the model
        lambda_ (scalar): Regularization parameter. Must be between 0..1. 
                         Determinate the weight of the regularization.
    """
    def __init__(self, x, y, w, b, lambda_):
        super().__init__(x, y, w, b, lambda_)

    def sigmoid(self, z):
        """
        Calcula la función sigmoide.
        
        Args:
            z (ndarray or scalar): valor de entrada
        
        Returns:
            g (ndarray or scalar): sigmoid(z) = 1 / (1 + e^(-z))
        """
        g = 1 / (1 + np.exp(-z))
        return g

    def f_w_b(self, x):
        """
        Calcula la predicción de regresión logística.
        Aplica la función sigmoide a la combinación lineal.
        
        Args:
            x (ndarray): Shape (m,n) características de entrada
        
        Returns:
            predictions (ndarray): Shape (m,) predicciones entre 0 y 1
        """
        z = x @ self.w + self.b
        predictions = self.sigmoid(z)
        return predictions

    def compute_cost(self):
        """
        Calcula el coste de la regresión logística con regularización L2.
        
        Función de coste:
        J(w,b) = -1/m * sum[y*log(y') + (1-y)*log(1-y')] + regularización
        
        Returns:
            total_cost (float): El coste de usar w,b como parámetros
        """
        m = len(self.y)  # Número de ejemplos
        
        # Calcular predicciones usando sigmoide
        predictions = self.f_w_b(self.x)
        
        # Evitar log(0) añadiendo epsilon
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Calcular coste logístico
        cost = -1/m * np.sum(self.y * np.log(predictions) + (1 - self.y) * np.log(1 - predictions))
        
        # Añadir regularización L2 si lambda_ > 0
        if self.lambda_ > 0:
            cost += self._regularizationL2Cost()
        
        return cost
    
    def compute_gradient(self):
        """
        Calcula el gradiente para regresión logística con regularización L2.
        
        El gradiente es:
        dj_dw = 1/m * X^T @ (predictions - y) + regularización
        dj_db = 1/m * sum(predictions - y)
        
        Returns:
            dj_dw (ndarray): Shape (n,) El gradiente del coste respecto a w
            dj_db (scalar): El gradiente del coste respecto a b
        """
        m = len(self.y)  # Número de ejemplos
        
        # Calcular predicciones usando sigmoide
        predictions = self.f_w_b(self.x)
        
        # Calcular errores
        errors = predictions - self.y
        
        # Calcular gradientes (versión vectorizada)
        dj_dw = (self.x.T @ errors) / m
        dj_db = np.sum(errors) / m
        
        # Añadir regularización L2 si lambda_ > 0
        if self.lambda_ > 0:
            dj_dw += self._regularizationL2Gradient()
        
        return dj_dw, dj_db


    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
