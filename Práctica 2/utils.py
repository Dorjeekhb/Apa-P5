import numpy as np
import pandas as pd


def cleanData(data):
    data["score"] = data["score"].apply(lambda x:  str(x).replace(",","."))
    data = data.drop(data[data["user score"] == "tbd"].index)
    data["user score"] = data["user score"].apply(lambda x:  str(x).replace(",","."))
    # Ambas columnas deben estar en la misma escala
    data["score"] = data["score"].astype(np.float64) / 10.0
    data["user score"] = data["user score"].astype(np.float64)
    return data

def cleanDataMulti(data):
    data = cleanData(data)
    data["critics"] = data["critics"].astype(np.float64)
    data["users"] = data["users"].astype(np.float64)
    return data

def load_data_csv(path,x_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    X = data[x_colum].to_numpy()
    y = data[y_colum].to_numpy()
    return X, y



def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    X_norm = np.zeros(X.shape)
    for i in range(X.shape[1]):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]
    
    return (X_norm, mu, sigma)

def load_data_csv_multi(path,x1_colum,x2_colum,x3_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanDataMulti(data)
    x1 = data[x1_colum].to_numpy()
    x2 = data[x2_colum].to_numpy()
    x3 = data[x3_colum].to_numpy()
    X = np.array([x1, x2, x3])
    X = X.T
    y = data[y_colum].to_numpy()
    return X, y

    
## 0 Malo, 1 Regular, 2 Notable, 3 Sobresaliente, 4 Must Play.
## 0 Malo, 1 Bueno
def load_data_csv_multi_logistic(path,x1_colum,x2_colum,x3_colum,y_colum):
    """
    Carga los datos para regresión logística.
    Transforma la salida a 0 o 1 basándose en si user score es < 7 (0=malo) o >= 7 (1=bueno)
    
    Args:
        path: ruta al archivo CSV
        x1_colum, x2_colum, x3_colum: nombres de las columnas de características
        y_colum: nombre de la columna de salida (user score)
    
    Returns:
        X: array de características (m, 3)
        y: array de etiquetas binarias (m,) - 0 para malo, 1 para bueno
    """
    X, y = load_data_csv_multi(path, x1_colum, x2_colum, x3_colum, y_colum)
    # Convertir a clases binarias: 0 si < 7 (malo), 1 si >= 7 (bueno)
    y = (y >= 7).astype(np.float64)
    return X, y


def GetNumGradientsSuccess(w1,w1Sol,b1,b1Sol):
    iterator = 0
    for i in range(len(w1)): 
        if np.isclose(w1[i],w1Sol[i]):
                iterator += 1
    if np.isclose(b1,b1Sol):
        iterator += 1
    return iterator