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

# =========================
# Matriz y métricas básicas
# =========================
def confusion_matrix(y_true, y_pred) -> np.ndarray:
    y_true = _as1d(y_true); y_pred = _as1d(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp],
                     [fn, tp]], dtype=int)

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pr(y_true, y_score, ax=None, label=None):
    """Traza PR y devuelve AUC-PR."""
    rec, prec = pr_curve_np(y_true, y_score)
    ap = auc_trapezoid(rec, prec)
    if ax is None: ax = plt.gca()
    ax.plot(rec, prec, lw=2, label=f"{label or 'PR'} (AUC-PR={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall")
    ax.grid(True); ax.legend()
    return ap

def plot_roc(y_true, y_score, ax=None, label=None):
    """Traza ROC y devuelve AUC-ROC."""
    fpr, tpr = roc_curve_np(y_true, y_score)
    ar = auc_trapezoid(fpr, tpr)
    if ax is None: ax = plt.gca()
    ax.plot(fpr, tpr, lw=2, label=f"{label or 'ROC'} (AUC={ar:.3f})")
    ax.plot([0,1],[0,1],'--',color='gray',lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR (Recall)"); ax.set_title("ROC")
    ax.grid(True); ax.legend()
    return ar

def plot_roc_pr(y_true, y_score):
    """Conveniencia: dibuja ambas curvas en figuras separadas y retorna (AUC-ROC, AUC-PR)."""
    # ROC
    fpr, tpr = roc_curve_np(y_true, y_score)
    auc_roc = auc_trapezoid(fpr, tpr)
    plt.figure(figsize=(4.2, 3.6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc_roc:.3f})")
    plt.tight_layout(); plt.show()
    # PR
    rec, prec = pr_curve_np(y_true, y_score)
    auc_pr = auc_trapezoid(rec, prec)
    plt.figure(figsize=(4.2, 3.6))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUC={auc_pr:.3f})")
    plt.tight_layout(); plt.show()
    return auc_roc, auc_pr

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

def tune_lambda(X_tr, y_tr, X_va, y_va, lambdas, lr=0.1, epochs=4000, tol=1e-6):
    best = dict(f1=-1, lam=None, thr=0.5, model=None)
    for lam in lambdas:
        m = LogisticRegressionL2(lam=lam, lr=lr, epochs=epochs, tol=tol, fit_intercept=True).fit(X_tr, y_tr)
        thr, f1 = best_threshold_for_f1(y_va, m.predict_proba(X_va))
        if f1 > best["f1"]:
            best.update(f1=f1, lam=lam, thr=thr, model=m)
    return best["model"], best["lam"], best["thr"], best["f1"]