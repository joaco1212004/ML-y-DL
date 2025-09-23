import numpy as np
import pandas as pd

# Columnas del dataset 
NUM_COLS = [
    "CellSize","CellShape","NucleusDensity","ChromatinTexture","CytoplasmSize",
    "CellAdhesion","MitosisRate","NuclearMembrane","GrowthFactor",
    "OxygenSaturation","Vascularization","InflammationMarkers",
]
CAT_COLS = ["CellType","GeneticMutation"]
TARGET   = "Diagnosis"

# rangos por dominio
RANGES = {
    "CellAdhesion": (0, 1),
    "NuclearMembrane": (1, 5),
    "OxygenSaturation": (0, 100),
    "Vascularization": (0, 10),
    "InflammationMarkers": (0, 100),
    "CellSize": (0, np.inf),
    "CytoplasmSize": (0, np.inf),
    "MitosisRate": (0, np.inf),
}

def to_nan_out_of_range(df: pd.DataFrame, ranges=RANGES) -> pd.DataFrame:
    """Convierte a NaN valores fuera de rango (o imposibles) en columnas numéricas."""
    df = df.copy()

    for c,(lo,hi) in ranges.items():
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")
            s[(s < lo) | (s > hi)] = np.nan
            df[c] = s
            
    return df

def split_train_val(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, stratify: bool = True):
    """Devuelve índices de train/val (opcionalmente estratificados por TARGET)."""
    n = len(df)
    rng = np.random.default_rng(seed)

    if stratify:
        y = df[TARGET].astype(int).to_numpy()
        idx0, idx1 = np.where(y==0)[0], np.where(y==1)[0]
        rng.shuffle(idx0); rng.shuffle(idx1)
        cut0, cut1 = int((1-test_size)*len(idx0)), int((1-test_size)*len(idx1))
        train_idx = np.r_[idx0[:cut0], idx1[:cut1]]
        val_idx   = np.r_[idx0[cut0:],  idx1[cut1:]]
        rng.shuffle(train_idx); rng.shuffle(val_idx)

        return train_idx, val_idx
    else:
        idx = np.arange(n); rng.shuffle(idx)
        cut = int((1-test_size) * n)

        return idx[:cut], idx[cut:]

def _clean_categories(s: pd.Series) -> pd.Series:
    return s.fillna("Unknown").replace("???", "Unknown").astype("string")

def fit_preprocessor(train_df: pd.DataFrame):
    """Calcula parámetros de imputación y escalado a partir de TRAIN."""
    num_median = train_df[NUM_COLS].apply(pd.to_numeric, errors="coerce").median()
    tmp = train_df.copy()

    for c in CAT_COLS:
        if c in tmp:
            tmp[c] = _clean_categories(tmp[c])

    X_train_oh = pd.get_dummies(tmp.drop(columns=[TARGET]), columns=CAT_COLS, drop_first=True)
    mu = X_train_oh.mean()
    sd = X_train_oh.std().replace(0, 1.0)
    params = dict(num_median=num_median, oh_columns=X_train_oh.columns, mu=mu, sd=sd)

    return params

def transform(df: pd.DataFrame, params):
    """Aplica imputación (mediana TRAIN), one-hot (columnas TRAIN) y z-score (μ,σ TRAIN)."""
    df = df.copy()

    # categóricas
    for c in CAT_COLS:
        if c in df:
            df[c] = _clean_categories(df[c])

    # imputación numérica
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce").fillna(params["num_median"])

    # dummies + reindex
    X = pd.get_dummies(df.drop(columns=[TARGET]), columns=CAT_COLS, drop_first=True)
    X = X.reindex(columns=params["oh_columns"], fill_value=0)

    # z-score
    X = (X - params["mu"]) / params["sd"]
    y = df[TARGET].astype(int).to_numpy()

    return X, y

def preprocess_train_val(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, stratify: bool = True):
    """Pipeline completo: split + fit preproc en TRAIN + transform a TRAIN/VAL."""
    train_idx, val_idx = split_train_val(df, test_size=test_size, seed=seed, stratify=stratify)
    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    params = fit_preprocessor(train_df)
    X_train, y_train = transform(train_df, params)
    X_val,   y_val   = transform(val_df,   params)

    return X_train.to_numpy(), y_train, X_val.to_numpy(), y_val, params