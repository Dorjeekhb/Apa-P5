import numpy as np
import copy
import math

from LinearRegression import LinearReg

class LinearRegMulti(LinearReg):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y, w, b, lambda_):
        super().__init__(x, y, w, b)
        self.lambda_ = lambda_

    def f_w_b(self, x):
        ret = x @ self.w + self.b
        return ret

    """
    Computes the cost function for linear regression with regularization.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    def compute_cost(self):
        n = len(self.y)  # Numero de ejemplos
        
        # Implementación para el coste
        predictions = self.f_w_b(self.x)
        cost = np.sum((predictions - self.y) ** 2) / (2 * n)
        
        # Añadir regularización si lambda_ > 0
        if self.lambda_ > 0:
            cost += self._regularizationL2Cost()
        
        return cost
    
    """
    Computes the gradient for linear regression with regularization
    
    Returns
      dj_dw (ndarray): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    """
    def compute_gradient(self):
        n = len(self.y)  # Numero de ejemplos
        
        # Implementación vectorizada
        predictions = self.f_w_b(self.x)
        errors = predictions - self.y
        
        dj_dw = (self.x.T @ errors) / n
        dj_db = np.sum(errors) / n
        
        # Añadir regularización si lambda_ > 0
        if self.lambda_ > 0:
            dj_dw += self._regularizationL2Gradient()
        
        return dj_dw, dj_db
    
    """
    Compute the regularization cost (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Cost (float): the regularization value of the current model
    """
    
    def _regularizationL2Cost(self):
        reg_cost_final = (self.lambda_ / (2 * len(self.y))) * np.sum(self.w ** 2)
        return reg_cost_final
    
    """
    Compute the regularization gradient (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Gradient (vector size n): the regularization gradient of the current model
    """ 
    
    def _regularizationL2Gradient(self):
        n = len(self.y)  # Numero de ejemplos
        reg_gradient_final = (self.lambda_ / n) * self.w
        return reg_gradient_final
        
    def gradient_descent(self, alpha, num_iters):
        """
        Performs batch gradient descent to learn theta. Updates theta by taking 
        num_iters gradient steps with learning rate alpha
        
        Args:
            alpha : (float) Learning rate
            num_iters : (int) number of iterations to run gradient descent
        Returns:
            w : (ndarray): Updated values of parameters of the model after running gradient descent
            b : (scalar) Updated value of parameter of the model after running gradient descent
            J_history : (ndarray): Shape (num_iters,) J at each iteration, primarily for graphing later
            w_initial : (ndarray): initial w value before running gradient descent
            b_initial : (scalar) initial b value before running gradient descent
        """
        # An array to store cost J and w's at each iteration — primarily for graphing later
        J_history = []
        w_history = []
        w_initial = copy.deepcopy(self.w)  # avoid modifying global w within function
        b_initial = copy.deepcopy(self.b)  # avoid modifying global b within function
        
        for i in range(num_iters):
            # Calculo de gradiantes
            dj_dw, dj_db = self.compute_gradient()
            self.w -= alpha * dj_dw
            self.b -= alpha * dj_db
            
            # Coste
            cost = self.compute_cost()
            J_history.append(cost)
            w_history.append(self.w)
        
        return self.w, self.b, J_history, w_initial, b_initial
    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
