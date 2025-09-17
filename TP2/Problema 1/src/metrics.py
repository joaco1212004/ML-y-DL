import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np    

def confusion_matrix_np(y_true, y_pred):
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    tp = int(((y_true==1)&(y_pred==1)).sum())
    return np.array([[tn, fp],[fn, tp]])

def basic_metrics(y_true, y_pred):
    cm = confusion_matrix_np(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp+tn)/cm.sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return cm, acc, prec, rec, f1

# curvas ROC/PR con acumulados
def roc_curve_np(y_true, y_score):
    order = np.argsort(-y_score)               # desc
    y = y_true[order]
    tps = np.cumsum(y==1)
    fps = np.cumsum(y==0)
    P = (y_true==1).sum()
    N = (y_true==0).sum()
    TPR = tps / P
    FPR = fps / N
    # agregar (0,0) y (1,1)
    FPR = np.r_[0.0, FPR, 1.0]
    TPR = np.r_[0.0, TPR, 1.0]
    return FPR, TPR

def pr_curve_np(y_true, y_score):
    order = np.argsort(-y_score)
    y = y_true[order]
    tps = np.cumsum(y==1)
    fps = np.cumsum(y==0)
    P = (y_true==1).sum()
    recall = tps / P
    precision = tps / (tps + fps + 1e-12)
    # agregar punto inicial (recall=0, precision=positives_rate)
    pos_rate = P / len(y_true)
    recall = np.r_[0.0, recall]
    precision = np.r_[pos_rate, precision]
    return recall, precision

def auc_trapezoid(x, y):
    return float(np.trapz(y, x))

def plot_confusion(cm, title="Matriz de confusión"):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
    plt.title(title); plt.tight_layout(); plt.show()

def plot_roc_pr(y_true, y_score):
    # ROC
    fpr, tpr = roc_curve_np(y_true, y_score)
    auc_roc = auc_trapezoid(fpr, tpr)
    plt.figure(figsize=(4.2,3.6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1],"--",color="gray",lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc_roc:.3f})")
    plt.tight_layout(); plt.show()
    # PR
    rec, prec = pr_curve_np(y_true, y_score)
    auc_pr = auc_trapezoid(rec, prec)
    plt.figure(figsize=(4.2,3.6))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUC={auc_pr:.3f})")
    plt.tight_layout(); plt.show()
    return auc_roc, auc_pr