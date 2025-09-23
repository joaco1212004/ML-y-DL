import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import LogisticRegressionL2

# =========================
# Helpers
# =========================
def _as1d(a) -> np.ndarray:
    """Asegura vector 1D para comparaciones y acumulados."""
    a = np.asarray(a)
    
    return a.reshape(-1)

def threshold_labels(y_score, thr: float = 0.5) -> np.ndarray:
    """Convierte puntajes/probabilidades en etiquetas {0,1} con un umbral dado."""
    return (np.asarray(y_score) >= thr).astype(int)

def undersample_majority(X, y, seed=42):
    """Baja la clase mayoritaria para empatar con la minoritaria."""
    rng = np.random.default_rng(seed)
    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        return X, y
    
    if len(idx0) > len(idx1):
        keep0 = rng.choice(idx0, size=len(idx1), replace=False)
        new_idx = np.r_[keep0, idx1]
    else:
        keep1 = rng.choice(idx1, size=len(idx0), replace=False)
        new_idx = np.r_[idx0, keep1]

    rng.shuffle(new_idx)

    return X[new_idx], y[new_idx]

def oversample_duplicate(X, y, seed=42):
    """Duplica aleatoriamente la clase minoritaria hasta igualar a la mayoritaria."""
    rng = np.random.default_rng(seed)
    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    n0, n1 = len(idx0), len(idx1)

    if n0 == 0 or n1 == 0:
        return X, y
    
    if n0 > n1:
        need = n0 - n1
        add = rng.choice(idx1, size=need, replace=True)
        new_idx = np.r_[np.arange(len(y)), add]
    else:
        need = n1 - n0
        add = rng.choice(idx0, size=need, replace=True)
        new_idx = np.r_[np.arange(len(y)), add]

    rng.shuffle(new_idx)

    return X[new_idx], y[new_idx]

def smote_simple(X, y, seed=42):
    """
    SMOTE básico (sin sklearn):
    para cada muestra minoritaria se elige al azar otra minoritaria y se interpola.
    Genera sintéticos hasta igualar a la mayoritaria.
    """
    rng = np.random.default_rng(seed)
    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    n0, n1 = len(idx0), len(idx1)

    if n0 == 0 or n1 == 0 or X.shape[0] < 2:
        return X, y

    # definimos minoritaria/mayoritaria
    if n0 > n1:
        min_idx, maj_n = idx1, n0
    else:
        min_idx, maj_n = idx0, n1

    gap = maj_n - len(min_idx)

    if gap <= 0:
        return X, y

    # sintetizamos 'gap' puntos
    X_min = X[min_idx]
    new_samples = []

    for _ in range(gap):
        i = rng.integers(0, len(X_min))
        j = rng.integers(0, len(X_min))

        while j == i and len(X_min) > 1:
            j = rng.integers(0, len(X_min))

        alpha = rng.random()
        x_new = X_min[i] + alpha * (X_min[j] - X_min[i])
        new_samples.append(x_new)

    X_syn = np.vstack(new_samples)
    y_syn = np.full((gap,), y[min_idx[0]])  # etiqueta de la minoritaria

    X_bal = np.vstack([X, X_syn])
    y_bal = np.concatenate([y, y_syn])
    idx = rng.permutation(len(y_bal))

    return X_bal[idx], y_bal[idx]

def eval_and_store(name, model, Xv, yv, thr):
    results = []
    pr_curves = []
    roc_curves = []

    y_score = model.predict_proba(Xv)
    y_pred  = (y_score >= thr).astype(int)
    cm, acc, pre, rec, f1 = basic_metrics(yv, y_pred)
    auc_pr  = auc_pr_np(yv, y_score)
    auc_roc = auc_roc_np(yv, y_score)
    results.append([name, acc, pre, rec, f1, auc_roc, auc_pr])
    rec_curve, prec_curve = pr_curve_np(yv, y_score)
    fpr_curve, tpr_curve  = roc_curve_np(yv, y_score)
    pr_curves.append((name, rec_curve, prec_curve))
    roc_curves.append((name, fpr_curve, tpr_curve))

    return results, pr_curves, roc_curves

# =========================
# Matriz y métricas básicas
# =========================
def confusion_matrix(y_true, y_pred) -> np.ndarray:
    y_true = _as1d(y_true); y_pred = _as1d(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return np.array([[tn, fp], [fn, tp]], dtype=int)

def accuracy(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    return float((cm[0,0] + cm[1,1]) / cm.sum())

def precision(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1,1], cm[0,1]
    return float(tp / (tp + fp)) if (tp + fp) else 0.0

def recall(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1,1], cm[1,0]

    return float(tp / (tp + fn)) if (tp + fn) else 0.0

def f1_score(y_true, y_pred) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return float(2 * p * r / (p + r)) if (p + r) else 0.0

def basic_metrics(y_true, y_pred):
    """Devuelve (cm, accuracy, precision, recall, f1)."""
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec  = recall(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    return cm, acc, prec, rec, f1

# =========================
# Curvas ROC / PR (sin sklearn)
# =========================
def roc_curve_np(y_true, y_score):
    """
    Retorna FPR, TPR construidos ordenando por score descendente.
    """
    y_true = _as1d(y_true); y_score = _as1d(y_score)
    order = np.argsort(-y_score)   # descendente
    y = y_true[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()

    TPR = tps / (P + 1e-12)
    FPR = fps / (N + 1e-12)

    # extremos (0,0) y (1,1)
    FPR = np.r_[0.0, FPR, 1.0]
    TPR = np.r_[0.0, TPR, 1.0]

    return FPR, TPR

def pr_curve_np(y_true, y_score):
    """
    Retorna recall, precision ordenando por score descendente.
    """
    y_true = _as1d(y_true); y_score = _as1d(y_score)
    order = np.argsort(-y_score)
    y = y_true[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    P = (y_true == 1).sum()

    recall = tps / (P + 1e-12)
    precision = tps / (tps + fps + 1e-12)

    # punto inicial: recall=0, precision=tasa de positivos
    pos_rate = P / len(y_true)
    recall = np.r_[0.0, recall]
    precision = np.r_[pos_rate, precision]

    return recall, precision

def auc_trapezoid(x, y) -> float:
    """Área bajo la curva por la regla del trapecio."""
    x = _as1d(x); y = _as1d(y)
    idx = np.argsort(x)
    return float(np.trapz(y[idx], x[idx]))

def auc_roc_np(y_true, y_score) -> float:
    fpr, tpr = roc_curve_np(y_true, y_score)
    return auc_trapezoid(fpr, tpr)

def auc_pr_np(y_true, y_score) -> float:
    rec, prec = pr_curve_np(y_true, y_score)
    return auc_trapezoid(rec, prec)

# =========================
# Plots
# =========================
def plot_confusion(cm, title="Matriz de confusión"):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pr(y_true, y_score, ax=None, label=None):
    """Traza PR y devuelve AUC-PR."""
    rec, prec = pr_curve_np(y_true, y_score)
    ap = auc_trapezoid(rec, prec)

    if ax is None: ax = plt.gca()

    ax.plot(rec, prec, lw=2, label=f"{label or 'PR'} (AUC-PR={ap:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall")
    ax.grid(True); ax.legend()

    return ap

def plot_roc(y_true, y_score, ax=None, label=None):
    """Traza ROC y devuelve AUC-ROC."""
    fpr, tpr = roc_curve_np(y_true, y_score)
    ar = auc_trapezoid(fpr, tpr)

    if ax is None: ax = plt.gca()

    ax.plot(fpr, tpr, lw=2, label=f"{label or 'ROC'} (AUC={ar:.4f})")
    ax.plot([0,1],[0,1],'--',color='gray',lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR (Recall)"); ax.set_title("ROC")
    ax.grid(True); ax.legend()

    return ar

# =========================
# Umbral óptimo por F1
# =========================
def best_threshold_for_f1(y_true, y_score, grid=None):
    """
    Busca el umbral que maximiza F1 sobre un grid.
    Devuelve (umbral, f1_max).
    """
    y_true = _as1d(y_true); y_score = _as1d(y_score)

    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)

    best_f1, best_thr = -1.0, 0.5

    for thr in grid:
        y_pred = threshold_labels(y_score, thr)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1

def tune_lambda(X_tr, y_tr, X_va, y_va, lambdas, *, sample_weight=None, lr=0.1, epochs=4000, tol=1e-6, bias=True, **fit_kwargs):
    best = (-1.0, None, 0.5, None)  # f1, lam, thr, model

    for lam in lambdas:
        m = LogisticRegressionL2(lam=lam, lr=lr, epochs=epochs, tol=tol, bias=bias, **fit_kwargs).fit(X_tr, y_tr, sample_weight=sample_weight)
        y_score = m.predict_proba(X_va)
        thr, f1 = best_threshold_for_f1(y_va, y_score)

        if f1 > best[0]:
            best = (f1, lam, thr, m)

    f1, lam, thr, m = best

    return m, lam, thr, f1