import numpy as np
import math

class MLP:
    """
    Constructor: Computes MLP.

    Args:
        inputLayer (int): size of input
        hiddenLayer (int): size of hidden layer.
        outputLayer (int): size of output layer
        seed (scalar): seed of the random numeric.
        epislom (scalar) : random initialization range. e.j: 1 = [-1..1], 2 = [-2,2]...
    """

    def __init__(self,inputLayer,hidenLayer, outputLayer, seed=0, epislom = 0.12):
        np.random.seed(seed)
        self.inputLayer = inputLayer
        self.hiddenLayer = hidenLayer
        self.outputLayer = outputLayer
        self.eps = epislom

        # Random init in [-eps, eps]
        self.theta1 = np.random.uniform(-self.eps, self.eps, size=(self.hiddenLayer, self.inputLayer + 1))
        self.theta2 = np.random.uniform(-self.eps, self.eps, size=(self.outputLayer, self.hiddenLayer + 1))

    def new_trained(self,theta1,theta2):
        self.theta1 = theta1
        self.theta2 = theta2

    def _size(self,x):
        return x.shape[0]

    def _sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoidPrime(self,a):
        return a * (1.0 - a)

    def feedforward(self,x):
        m = x.shape[0]
        a1 = np.concatenate([np.ones((m,1)), x], axis=1)          # (m, n+1)
        z2 = a1 @ self.theta1.T                                   # (m, hidden)
        a2 = self._sigmoid(z2)                                     # (m, hidden)
        a2b = np.concatenate([np.ones((m,1)), a2], axis=1)        # (m, hidden+1)
        z3 = a2b @ self.theta2.T                                  # (m, output)
        a3 = self._sigmoid(z3)                                     # (m, output)
        return a1, a2, a3, z2, z3

    def compute_cost(self, yPrime,y, lambda_):
        m = y.shape[0]
        eps = 1e-12
        term1 = -y * np.log(yPrime + eps)
        term2 = -(1 - y) * np.log(1 - yPrime + eps)
        J = (1.0 / m) * np.sum(term1 + term2)
        reg = self._regularizationL2Cost(m, lambda_)
        return J + reg

    def predict(self,a3):
        # Devuelve etiquetas 1..10, pero mapea 10 -> 0 para coincidir con el ground truth normalizado
        p = np.argmax(a3, axis=1) + 1
        p[p == 10] = 0
        return p

    def compute_gradients(self, x, y, lambda_):
        m = x.shape[0]
        a1, a2, a3, z2, z3 = self.feedforward(x)
        J = self.compute_cost(a3, y, lambda_)

        # Backprop
        delta3 = a3 - y                                   # (m, output)
        a2b = np.concatenate([np.ones((a2.shape[0],1)), a2], axis=1)
        delta2 = (delta3 @ self.theta2)[:,1:] * self._sigmoidPrime(a2)

        Delta1 = delta2.T @ a1                            # (hidden, n+1)
        Delta2 = delta3.T @ a2b                           # (output, hidden+1)

        reg1 = self._regularizationL2Gradient(self.theta1, lambda_, m)
        reg2 = self._regularizationL2Gradient(self.theta2, lambda_, m)

        grad1 = (1.0/m) * Delta1 + reg1
        grad2 = (1.0/m) * Delta2 + reg2

        return (J, grad1, grad2)

    def _regularizationL2Gradient(self, theta, lambda_, m):
        reg = (lambda_ / m) * theta
        reg[:,0] = 0.0
        return reg

    def _regularizationL2Cost(self, m, lambda_):
        t1 = self.theta1[:,1:]
        t2 = self.theta2[:,1:]
        return (lambda_/(2.0*m)) * (np.sum(t1*t1) + np.sum(t2*t2))

    def backpropagation(self, x, y, alpha, lambda_, numIte, verbose=0):
        Jhistory = []
        for i in range(numIte):
            J, grad1, grad2 = self.compute_gradients(x, y, lambda_)
            self.theta1 -= alpha * grad1
            self.theta2 -= alpha * grad2
            Jhistory.append(float(J))
            if verbose > 0 :
                if i % verbose == 0 or i == (numIte-1):
                    print(f"Iteration {(i+1):6}: Cost {float(J):8.6f}")
        return Jhistory


def target_gradient(input_layer_size,hidden_layer_size,num_labels,x,y,reg_param):
    mlp = MLP(input_layer_size,hidden_layer_size,num_labels)
    J, grad1, grad2 = mlp.compute_gradients(x,y,reg_param)
    return J, grad1, grad2, mlp.theta1, mlp.theta2

def costNN(Theta1, Theta2,x, ys, reg_param):
    mlp = MLP(x.shape[1],1, ys.shape[1])
    mlp.new_trained(Theta1,Theta2)
    J, grad1, grad2 = mlp.compute_gradients(x,ys,reg_param)
    return J, grad1, grad2

def MLP_backprop_predict(X_train,y_train, X_test, alpha, lambda_, num_ite, verbose):
    mlp = MLP(X_train.shape[1],25,y_train.shape[1])
    mlp.backpropagation(X_train,y_train,alpha,lambda_,num_ite,verbose)
    a3 = mlp.feedforward(X_test)[2]
    y_pred = mlp.predict(a3)
    return y_pred
