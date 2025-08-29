import pandas as pd
import numpy as np

class RegresionLineal:
    """
    Regresión lineal OLS con dos modos de entrenamiento:
      - fit_pinv(): solución cerrada con pseudo-inversa
      - fit_gd():   descenso por gradiente batch para MSE

    Guarda los pesos en self.coef (incluye intercepto si add_intercept=True).
    """

    def __init__(self, x, y, bias: bool = True, l1: float = 0.0, l2: float = 0.0):
        self.feature_names = None
        self.l1 = l1
        self.l2 = l2

        if isinstance(x, pd.DataFrame):
            self.feature_names = list(x.columns)
            x = x.to_numpy()
        else:
            x = np.asarray(x)

        y = (y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)).reshape(-1) # aceptar DataFrame/Series o arrays

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        self.bias = bias
        self.x = self._add_bias(x) if bias else x
        self.y = y

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("Las dimensiones de x e y no coinciden")
        
        if self.feature_names == None:
            n_features = self.x.shape[1] - (1 if bias else 0)
            self.feature_names = [f"x{i}" for i in range(n_features)]
        self.colnames_ = (["bias"] if bias else []) + self.feature_names

        self.coef = None
    
    def _add_bias(self, x: np.ndarray) -> np.ndarray:
        return np.c_[np.ones((x.shape[0], 1)), x]
    
    def _prepare_x (self, newX):
        xn = (newX.to_numpy() if isinstance(newX, pd.DataFrame) else np.asarray(newX)) # acepta pandas o numpy; preserva forma correcta
        if xn.ndim == 1:
            xn = xn.reshape(-1, 1)
        return self._add_bias(xn) if self.bias else xn

    @staticmethod
    def mse_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return np.mean((y_true - y_pred) ** 2)
    
    def mse (self, x_test = None, y_test = None):
        if self.coef is None:
            raise RuntimeError("Modelo no entrenado")
    
        if x_test is None or y_test is None:
            y_pred = self.x @ self.coef
            y_true = self.y
        else:
            y_pred = self.predict(x_test)
            y_true = y_test.to_numpy() if isinstance(y_test, (pd.Series, pd.DataFrame)) else np.asarray(y_test).reshape(-1)   
        return self.mse_from_pred(y_true, y_pred)
    
    def fit_pinv(self):
        """Solución cerrada: w = pinv(X) @ y"""

        self.coef = np.linalg.pinv(self.x) @ self.y
        return self

    def fit_gd(self, lr: float = 0.01, epochs: int = 10000, tol: float = 1e-6, verbose: bool = False):
        """
        Descenso por gradiente batch para minimizar MSE.
        Actualización: w <- w - lr * (2/n) * X^T (Xw - y)
        """

        n, d = self.x.shape
        w = np.zeros(d)
        prev_loss = np.inf

        for i in range (1, epochs + 1):
            y_hat = self.x @ w
            grad = (2/n) * (self.x.T @ (y_hat - self.y))
            w -= lr * grad

            if i % 200 == 0 or i == epochs:
                loss = np.mean((y_hat - self.y) ** 2)
                if verbose:
                    print(f"iter {i}, loss(mse): {loss:.4f}")
                
                if abs(prev_loss - loss) < tol:
                    if verbose:
                        print(f"Converged at iter {i}, loss(mse): {loss:.4f}")
                    break
                prev_loss = loss
            
        self.coef = w
        return self

    def predict (self, newX):
        if self.coef is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() antes de predecir.")
        xp = self._prepare_x(newX) 
        return xp @ self.coef
    
    @staticmethod
    def score_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1 - ss_res / ss_tot)
    
    def score (self, x_test = None, y_test = None):
        """R² = 1 - SSE/SST."""

        if x_test is None or y_test is None:
            y_pred = self.x @ self.coef
            y_true = self.y
        else:
            y_pred = self.predict(x_test)
            y_true = y_test.to_numpy() if isinstance(y_test, (pd.Series, pd.DataFrame)) else np.asarray(y_test).reshape(-1)   
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def print_coef(self, dec: int = 5):
        """
        Imprime los coeficientes alineados con sus nombres de variables.
        """

        if self.coef is None:
            print("Modelo no entrenado")
            return
        ancho = max(len(name) for name in self.colnames_) + 2
        for name, coef in zip(self.colnames_, self.coef):
            print(f"{name: <{ancho}}: {coef:.{dec}f}")

    @staticmethod
    def standard_scale (x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estandariza columnas: (X - mean) / std. Devuelve (Xz, mean, std).
        Útil antes de fit_gd cuando hay escalas muy distintas.
        """
        
        x = np.asarray(x, type = float)
        mu = x.mean(axis = 0)
        sd = x.std(axis = 0, ddof = 0)
        sd[sd == 0] = 1.0
        return (x - mu) / sd, mu, sd
    
    @staticmethod
    def minmax_scale (x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Escala a [0, 1] por columna. Devuelve (Xmm, min, max)."""

        x = np.asarray(x, type = float)
        minx = x.min(axis = 0)
        maxx = x.max(axis = 0)
        range_x = maxx - minx
        range_x[range_x == 0] = 1.0
        return (x - minx) / range_x, minx, range_x
    
    @staticmethod
    def _soft_threshold(z: np.ndarray, t: float) -> np.ndarray:
        # operador proximal de L1: soft-thresholding
        return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

    def fit_ridge(self, l2: float):
        """Ridge regression (L2 regularization)"""
        n_features = self.x.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # no regularizar el bias
        self.coef = np.linalg.inv(self.x.T @ self.x + l2 * I) @ (self.x.T @ self.y)
        return self
    
    def fit_lasso(self, l1: float, lr=0.01, epochs=1000):
        """Lasso regression (L1 regularization) usando GD"""
        n, m = self.x.shape
        w = np.zeros(m)

        for _ in range(epochs):
            grad = (2/n) * self.x.T @ (self.x @ w - self.y) + l1 * np.sign(w)
            w -= lr * grad
        
        self.coef = w
        return self