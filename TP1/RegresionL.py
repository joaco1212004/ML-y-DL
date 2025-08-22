import pandas as pd
import numpy as np

class RegresionLineal:
    def __init__ (self, x, y, bias: bool = True):
        self.feature_names = None

        if isinstance(x, pd.DataFrame):
            self.feature_names = list(x.columns)
            x = x.to_numpy()
        else:
            x = np.asarray(x)

        y = (y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)).reshape(-1)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        self.bias = bias
        self.x = self._add_bias(x) if bias else x
        self.y = y

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("Las dimensiones de x e y no coinciden")
        
        if self.feature_names == None:
            self.feature_names = [f"x{i}" for i in range(self.x.shape[1]) - (1 if bias else 0)]
        self.colnames_ = (["bias"] if bias else []) + self.feature_names

        self.coef: np.ndarray | None = None
    
    def _add_bias(x: np.ndarray) -> np.ndarray:
        return np.c_[np.ones((x.shape[0], 1)), x]
    
    def preprare_x (self, newX):
        xn = (newX.to_numpy() if isinstance(newX, pd.DataFrame) else np.asarray(newX))
        if xn.ndim == 1:
            xn = xn.reshape(-1, 1)
        return self._add_bias(xn) if self.bias else xn

    @staticmethod
    def mse_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def mse (self, x_test = None, y_test = None):
        if x_test == None or y_test == None:
            y_pred = self.x @ self.coef
            y_true = self.y
        else:
            y_pred = self.predict(x_test)
            y_true = y_test.to_numpy() if isinstance(y_test, (pd.Series, pd.DataFrame)) else np.asarray(y_test).reshape(-1)   
        return self.mse_from_pred(y_true, y_pred)
    
    def fit_pinv(self):
        self.coef = np.linalg.pinv(self.x) @ self.y
        return self

    def fit_gd(self):