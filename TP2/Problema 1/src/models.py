import numpy as np

class LogisticRegressionL2:
    def __init__(self, lam=0.0, lr=0.1, epochs=4000, tol=1e-6, bias=True, verbose=False):
        self.lam = float(lam)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.tol = float(tol)
        self.bias = bool(bias)
        self.verbose = bool(verbose)
        self.w = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        # Evita overflow en exp()
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        # <-- CAST EXPLÍCITO A FLOAT PARA EVITAR dtype=object
        X = np.asarray(X, dtype=float)
        z = X @ self.w + (self.b if self.bias else 0.0)
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= float(threshold)).astype(int)

    def fit(self, X, y, sample_weight=None, max_grad_norm=10.0, max_backoff=5):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n, d = X.shape

        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0

        if sample_weight is None:
            sw = np.ones(n, dtype=np.float64)
        else:
            sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)

        sw_sum = float(sw.sum()) if sw.size else 1.0
        if sw_sum <= 0:
            sw = np.ones(n, dtype=np.float64)
            sw_sum = float(n)

        prev_loss = np.inf
        lr = float(self.lr)  # copia local (podemos backoff)

        for it in range(1, self.epochs + 1):
            # forward
            z = X @ self.w + (self.b if self.bias else 0.0)
            # log-loss estable: log(1+exp(z)) - y*z
            logloss = (sw * (np.logaddexp(0.0, z) - y * z)).sum() / sw_sum
            reg = 0.5 * self.lam * np.dot(self.w, self.w)
            loss = float(logloss + reg)

            # si la loss no es finita, retrocede y baja lr
            backoff = 0
            while not np.isfinite(loss) and backoff < max_backoff:
                lr *= 0.5
                # achicamos un poco los parámetros para volver a región estable
                self.w *= 0.5
                self.b *= 0.5
                z = X @ self.w + (self.b if self.bias else 0.0)
                logloss = (sw * (np.logaddexp(0.0, z) - y * z)).sum() / sw_sum
                reg = 0.5 * self.lam * np.dot(self.w, self.w)
                loss = float(logloss + reg)
                backoff += 1
            if not np.isfinite(loss):
                # último recurso: reiniciar esta iter y seguir
                self.w[:] = 0.0
                self.b = 0.0
                lr = max(lr * 0.5, 1e-6)

            # probas seguras
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
            err = p - y

            # gradientes + L2 en w (no en bias)
            gw = (X.T @ (sw * err)) / sw_sum + self.lam * self.w
            gb = (sw * err).sum() / sw_sum if self.bias else 0.0

            # clipping de gradiente
            gnorm = float(np.linalg.norm(gw))
            if gnorm > max_grad_norm:
                gw *= (max_grad_norm / (gnorm + 1e-12))
            if self.bias:
                gb = float(np.clip(gb, -max_grad_norm, max_grad_norm))

            # paso
            self.w -= lr * gw
            if self.bias:
                self.b -= lr * gb

            # saneo por si acaso (evita NaN/Inf propagados)
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=1e6, neginf=-1e6)
            if self.bias:
                self.b = float(np.nan_to_num(self.b, nan=0.0, posinf=1e6, neginf=-1e6))

            # criterio de parada
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"converged @ iter {it}, loss={loss:.6f}")
                break
            prev_loss = loss

        return self