import numpy as np

class LogisticRegressionL2:
    def __init__(self, lam=0.0, lr=0.1, epochs=5000, tol=1e-6, fit_intercept=True, verbose=False):
        self.lam = float(lam)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.tol = float(tol)
        self.fit_intercept = bool(fit_intercept)
        self.verbose = verbose
        self.w = None

    @staticmethod
    def _sigmoid(z):
        # numerically stable
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_bias(self, X):
        if not self.fit_intercept: 
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def fit(self, X, y):
        Xb = self._add_bias(X)
        n, d = Xb.shape
        self.w = np.zeros(d, dtype=float)

        prev_loss = np.inf
        for i in range(1, self.epochs + 1):
            p = self._sigmoid(Xb @ self.w)             # (n,)
            # grad MSE-logistic = X^T (p - y)/n  +  lam * w  (no regularizar bias)
            grad = (Xb.T @ (p - y)) / n
            if self.fit_intercept:
                grad[0] += 0.0                 # bias sin L2
                grad[1:] += self.lam * self.w[1:]
            else:
                grad += self.lam * self.w

            self.w -= self.lr * grad

            # pérdida logística + L2 para early stopping
            if i % 200 == 0 or i == self.epochs:
                p_safe = np.clip(p, 1e-12, 1 - 1e-12)
                logloss = -np.mean(y*np.log(p_safe) + (1-y)*np.log(1-p_safe))
                reg = 0.5 * self.lam * (self.w[1:] @ self.w[1:]) if self.fit_intercept else 0.5 * self.lam * (self.w @ self.w)
                loss = logloss + reg
                if self.verbose:
                    print(f"iter {i:5d}  loss={loss:.6f}")
                if abs(prev_loss - loss) < self.tol:
                    if self.verbose:
                        print(f"Converged at iter {i}, loss={loss:.6f}")
                    break
                prev_loss = loss
        return self

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        return self._sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)