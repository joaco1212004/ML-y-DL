import numpy as np

class LDA:
    """Análisis Discriminante Lineal (pooled covariance)."""
    def __init__(self, reg=1e-5):
        self.reg = reg  # ridge para estabilidad numérica
        self.means_ = None
        self.cov_inv_ = None
        self.priors_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n, d = X.shape

        means = []
        priors = []
        # covarianza “pooled”
        S = np.zeros((d, d), float)
        for c in self.classes_:
            Xc = X[y == c]
            priors.append(len(Xc) / n)
            mu = Xc.mean(axis=0)
            means.append(mu)
            # suma de covarianzas con corrección de grados de libertad
            S += np.cov(Xc, rowvar=False, bias=False) * (len(Xc) - 1)

        S /= (n - K)
        S += self.reg * np.eye(d)
        self.cov_inv_ = np.linalg.inv(S)
        self.means_ = np.vstack(means)          # (K, d)
        self.priors_ = np.asarray(priors)       # (K,)
        return self

    def _discriminant(self, X):
        # δ_k(x) = x^T Σ^{-1} μ_k - 0.5 μ_k^T Σ^{-1} μ_k + log π_k
        X = np.asarray(X, float)
        A = self.cov_inv_ @ self.means_.T                    # (d,K)
        first = X @ A                                        # (n,K)
        second = 0.5 * np.sum(self.means_ @ self.cov_inv_ * self.means_, axis=1)  # (K,)
        return first - second + np.log(self.priors_)

    def predict_proba(self, X):
        g = self._discriminant(X)
        g -= g.max(axis=1, keepdims=True)  # estabilidad
        p = np.exp(g)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def predict(self, X):
        idx = np.argmax(self._discriminant(X), axis=1)
        return self.classes_[idx]

class SoftmaxRegression:
    """Logística multiclase (softmax) con L2 en W (no en el sesgo)."""
    def __init__(self, lam=0.0, lr=0.1, epochs=3000, tol=1e-6, bias=True, verbose=False):
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.bias = bias
        self.verbose = verbose
        self.W = None   # (d, K)
        self.b = None   # (K,)
        self.classes_ = None

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def predict_proba(self, X):
        Z = X @ self.W + (self.b if self.bias else 0.0)
        return self._softmax(Z)

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).reshape(-1)
        n, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        y_map = {c:i for i,c in enumerate(self.classes_)}
        t = np.array([y_map[yy] for yy in y])                # 0..K-1
        Y = np.eye(K)[t]                                     # one-hot (n,K)

        self.W = np.zeros((d, K), float)
        self.b = np.zeros(K, float)

        prev_loss = np.inf
        for it in range(1, self.epochs+1):
            P = self.predict_proba(X)                        # (n,K)
            # cross-entropy + L2
            eps = 1e-12
            loss = -np.sum(Y * np.log(P + eps)) / n + 0.5*self.lam*np.sum(self.W*self.W)
            # gradientes
            E = (P - Y) / n                                  # (n,K)
            gW = X.T @ E + self.lam * self.W                 # (d,K)
            gb = E.sum(axis=0) if self.bias else 0.0         # (K,)

            # paso
            self.W -= self.lr * gW
            if self.bias:
                self.b -= self.lr * gb

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"converged @ {it}, loss={loss:.6f}")
                break
            prev_loss = loss
        return self

def _entropy(counts):
    tot = counts.sum()
    if tot <= 0: return 0.0
    p = counts / tot
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

class _TreeNode:
    __slots__ = ("feat", "thr", "left", "right", "proba", "is_leaf")
    def __init__(self):
        self.feat = None
        self.thr = None
        self.left = None
        self.right = None
        self.proba = None
        self.is_leaf = True

class DecisionTreeEntropy:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # None|'sqrt'|int
        self.rng = np.random.default_rng(random_state)
        self.root = None
        self.classes_ = None

    def _best_split(self, X, y, feat_idx):
        # Devuelve (best_gain, best_thr) para una feature
        x = X[:, feat_idx]
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # candidatos de umbral: puntos medios donde cambia la clase
        dif = np.diff(y_sorted)
        idx_cuts = np.where(dif != 0)[0]
        if len(idx_cuts) == 0:
            return -1.0, None

        parent_counts = np.bincount(y, minlength=self.K)
        parent_H = _entropy(parent_counts)

        best_gain, best_thr = -1.0, None
        left_counts = np.zeros(self.K, int)
        right_counts = parent_counts.copy()

        for i in idx_cuts:
            cls = y_sorted[i]
            left_counts[cls] += 1
            right_counts[cls] -= 1

            nL = i + 1
            nR = len(y_sorted) - nL
            if nL < self.min_samples_split or nR < self.min_samples_split:
                continue

            H = (nL * _entropy(left_counts) + nR * _entropy(right_counts)) / (nL + nR)
            gain = parent_H - H
            if gain > best_gain:
                best_gain = gain
                best_thr = 0.5 * (x_sorted[i] + x_sorted[i+1])

        return best_gain, best_thr

    def _grow(self, X, y, depth):
        node = _TreeNode()
        counts = np.bincount(y, minlength=self.K)
        node.proba = (counts + 1e-12) / (counts.sum() + 1e-12)  # suavizado
        node.is_leaf = True

        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(np.unique(y)) == 1 or \
           X.shape[0] < self.min_samples_split:
            return node

        # elegir subconjunto de features
        d = X.shape[1]
        if self.max_features is None:
            m = d
        elif self.max_features == 'sqrt':
            m = max(1, int(np.sqrt(d)))
        elif isinstance(self.max_features, int):
            m = min(d, self.max_features)
        else:
            m = d
        feats = self.rng.choice(d, size=m, replace=False)

        # buscar mejor split en ese subconjunto
        best_gain, best_feat, best_thr = -1.0, None, None
        for f in feats:
            gain, thr = self._best_split(X, y, f)
            if gain > best_gain:
                best_gain, best_feat, best_thr = gain, f, thr

        if best_gain <= 1e-12 or best_thr is None:
            return node  # hoja

        # dividir y recursión
        node.is_leaf = False
        node.feat = best_feat
        node.thr = best_thr
        mask = X[:, best_feat] <= best_thr
        node.left  = self._grow(X[mask],  y[mask],  depth+1)
        node.right = self._grow(X[~mask], y[~mask], depth+1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        # mapear clases a 0..K-1 internamente
        self.classes_ = np.unique(y)
        self.K = len(self.classes_)
        y_map = {c:i for i,c in enumerate(self.classes_)}
        yi = np.array([y_map[yy] for yy in y], int)
        self.root = self._grow(X, yi, depth=0)
        return self

    def _proba_one(self, x, node):
        if node.is_leaf:
            return node.proba
        if x[node.feat] <= node.thr:
            return self._proba_one(x, node.left)
        else:
            return self._proba_one(x, node.right)

    def predict_proba(self, X):
        X = np.asarray(X, float)
        P = np.vstack([self._proba_one(x, self.root) for x in X])
        return P

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]


class RandomForest:
    def __init__(self, n_estimators=50, max_depth=None, min_samples_split=2,
                 max_features='sqrt', bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.rng = np.random.default_rng(random_state)
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        n = len(y)
        self.classes_ = np.unique(y)
        self.trees = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = self.rng.integers(0, n, size=n)
                Xi, yi = X[idx], y[idx]
            else:
                Xi, yi = X, y
            t = DecisionTreeEntropy(max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    max_features=self.max_features,
                                    random_state=int(self.rng.integers(0, 1e9)))
            t.fit(Xi, yi)
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        Ps = [t.predict_proba(X) for t in self.trees]   # lista (n,K)
        P = np.mean(Ps, axis=0)
        return P

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

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