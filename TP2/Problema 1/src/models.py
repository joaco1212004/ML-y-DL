import numpy as np

def _sigmoid(z):
    # numéricamente estable
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

class LogisticRegressionL2:
    """
    Regresión logística binaria con regularización L2 (Ridge).
    - No regulariza el término de sesgo (bias).
    """
    def __init__(self, lam: float = 0.0, lr: float = 0.1, epochs: int = 5000, tol: float = 1e-6,
                 bias: bool = True, verbose: bool = False):
        self.lam = float(lam)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.tol = float(tol)
        self.bias = bool(bias)
        self.verbose = bool(verbose)
        self.w = None

    def _add_bias(self, X):
        if not self.bias:
            return X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _bce_loss(self, y, p, w):
        # Binary cross-entropy + L2 (sin bias)
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        n = y.shape[0]
        ce = - (y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        if self.bias:
            w_reg = w[1:]
        else:
            w_reg = w
        reg = 0.5 * self.lam * np.dot(w_reg, w_reg) / n
        return ce + reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        Xb = self._add_bias(X)

        n, d = Xb.shape
        w = np.zeros(d, dtype=float)
        last = np.inf

        for i in range(1, self.epochs + 1):
            z = Xb @ w
            p = _sigmoid(z)

            # gradiente de BCE + L2 (sin bias)
            grad = (Xb.T @ (p - y)) / n
            if self.lam != 0.0:
                if self.bias:
                    grad[1:] += (self.lam / n) * w[1:]
                else:
                    grad += (self.lam / n) * w

            w -= self.lr * grad

            if i % 200 == 0 or i == self.epochs:
                loss = self._bce_loss(y, p, w)
                if self.verbose:
                    print(f"iter {i} | loss {loss:.6f}")
                if abs(last - loss) < self.tol:
                    break
                last = loss

        self.w = w
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        return _sigmoid(Xb @ self.w)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)