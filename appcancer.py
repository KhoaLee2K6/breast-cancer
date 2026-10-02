from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import io
import math
import numpy as np
import pandas as pd
import joblib
import uvicorn
import pandas as pd

# ML (survival dataset)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)


# ==============================================================
# 1. MODEL  —  graceful loading  (SVM 30 features - gốc)
# ==============================================================
MODEL_PATH = os.environ.get("BC_MODEL_PATH", "svm_model.pkl")

_model = None
_model_error: Optional[str] = None
try:
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        print(f"[OK] Loaded model: {MODEL_PATH}")
    else:
        _model_error = f"Model file not found: {MODEL_PATH}"
        print(f"[WARN] {_model_error}")
except Exception as exc:
    _model_error = str(exc)
    print(f"[WARN] Failed to load model: {exc}")


# ==============================================================
# 1b. SURVIVAL DATASET  (cho dashboard phân tích)
# ==============================================================
DATA_PATH = os.environ.get("BC_DATA_PATH", "breast_cancer.csv")


def _make_demo_dataframe(n: int = 1500) -> pd.DataFrame:
    """Sinh dữ liệu demo khi không tìm thấy file CSV."""
    rng = np.random.default_rng(42)
    races = np.array(["White", "Black", "Other"])
    marital = np.array(["Married", "Single", "Divorced", "Widowed", "Separated"])
    stages6 = np.array(["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
    diffs = np.array([
        "Well differentiated", "Moderately differentiated",
        "Poorly differentiated", "Undifferentiated",
    ])
    grades = np.array([1, 2, 3, 4])
    t_stages = np.array(["T1", "T2", "T3", "T4"])
    n_stages = np.array(["N1", "N2", "N3"])
    a_stages = np.array(["Regional", "Distant"])
    yn = np.array(["Positive", "Negative"])

    age = rng.integers(28, 92, n)
    race = rng.choice(races, n, p=[0.74, 0.16, 0.10])
    ms = rng.choice(marital, n, p=[0.55, 0.20, 0.12, 0.08, 0.05])
    t_stage = rng.choice(t_stages, n, p=[0.36, 0.40, 0.14, 0.10])
    n_stage = rng.choice(n_stages, n, p=[0.55, 0.30, 0.15])
    stage6 = rng.choice(stages6, n, p=[0.34, 0.30, 0.20, 0.11, 0.05])
    diff = rng.choice(diffs, n, p=[0.20, 0.45, 0.30, 0.05])
    grade = rng.choice(grades, n, p=[0.20, 0.45, 0.30, 0.05])
    a_stage = rng.choice(a_stages, n, p=[0.90, 0.10])
    tumor_size = np.clip(rng.normal(30, 22, n).astype(int), 1, 140)
    estrogen = rng.choice(yn, n, p=[0.64, 0.36])
    progesterone = rng.choice(yn, n, p=[0.60, 0.40])
    nodes_exam = rng.integers(1, 40, n)
    nodes_pos = np.clip((nodes_exam * rng.uniform(0, 0.6, n)).astype(int), 0, 25)
    survival = np.clip(rng.normal(55, 25, n).astype(int), 1, 120)

    risk = (
        0.02 * (age - 60)
        + 0.03 * (tumor_size - 30)
        + 0.20 * nodes_pos
        + 0.35 * (grade >= 3).astype(int)
        + 0.45 * (estrogen == "Negative").astype(int)
        + 0.30 * (progesterone == "Negative").astype(int)
        + 0.50 * (a_stage == "Distant").astype(int)
    )
    p_dead = 1.0 / (1.0 + np.exp(-(risk - 1.1)))
    status = np.where(rng.random(n) < p_dead, "Dead", "Alive")

    return pd.DataFrame({
        "age": age, "race": race, "marital_status": ms,
        "t_stage": t_stage, "n_stage": n_stage, "stage_6th": stage6,
        "differentiate": diff, "grade": grade, "a_stage": a_stage,
        "tumor_size": tumor_size,
        "estrogen_status": estrogen, "progesterone_status": progesterone,
        "regional_node_examined": nodes_exam,
        "regional_node_positive": nodes_pos,
        "survival_months": survival, "status": status,
    })


_df: Optional[pd.DataFrame] = None
_df_error: Optional[str] = None
try:
    if os.path.exists(DATA_PATH):
        _df = pd.read_csv(DATA_PATH)
        _df.columns = [str(c).strip().lower().replace(" ", "_") for c in _df.columns]
        _df = _df.loc[:, ~_df.columns.str.startswith("unnamed")]
        # chuẩn hoá tên cột về snake_case mong đợi
        rename_map = {
            "6th_stage": "stage_6th", "t_stage": "t_stage", "n_stage": "n_stage",
            "reginol_node_positive": "regional_node_positive",
            "regional_node_examined": "regional_node_examined",
            "survival_months": "survival_months",
        }
        _df = _df.rename(columns=rename_map)
        print(f"[OK] Loaded dataset: {DATA_PATH} shape={_df.shape}")
    else:
        _df = _make_demo_dataframe()
        print("[WARN] Dataset not found — using synthetic demo data.")
except Exception as exc:
    _df_error = str(exc)
    _df = _make_demo_dataframe()
    print(f"[WARN] Dataset load failed ({exc}) — using synthetic demo data.")

# Chuẩn hoá cột target nếu có
if _df is not None and "status" in _df.columns:
    _df["status"] = _df["status"].astype(str).str.strip().str.title()


# ==============================================================
# 1c. HELPERS  —  filters + ML training
# ==============================================================
CAT_COLS = [
    "race", "marital_status", "t_stage", "n_stage", "stage_6th",
    "differentiate", "a_stage", "estrogen_status", "progesterone_status",
]
NUM_COLS = [
    "age", "grade", "tumor_size",
    "regional_node_examined", "regional_node_positive", "survival_months",
]


def _apply_filters(df: pd.DataFrame, f: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not f:
        return df
    d = df.copy()

    # numeric ranges
    for col in NUM_COLS:
        lo = f.get(f"{col}_min")
        hi = f.get(f"{col}_max")
        if lo is not None:
            d = d[d[col] >= lo]
        if hi is not None:
            d = d[d[col] <= hi]

    # categorical multi-select
    for col in CAT_COLS + ["status"]:
        vals = f.get(col)
        if vals:
            d = d[d[col].isin(vals)]

    # free-text search
    q = (f.get("search") or "").strip().lower()
    if q:
        mask = np.zeros(len(d), dtype=bool)
        for c in d.columns:
            mask |= d[c].astype(str).str.lower().str.contains(q, na=False).to_numpy()
        d = d[mask]

    return d


def _safe(v):
    """Chuyển numpy/pandas scalar → JSON-safe."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def _prepare_xy(df: pd.DataFrame):
    """Trả về X (DataFrame đã encode), y (0/1), encoders, feature names."""
    if "status" not in df.columns:
        raise ValueError("Dataset thiếu cột 'status'.")

    df = df.copy()
    y_raw = df["status"].astype(str).str.lower()
    y = y_raw.isin(["dead", "1", "yes", "true"]).astype(int).to_numpy()

    feat_cols = [c for c in df.columns if c != "status"]
    X = pd.DataFrame(index=df.index)
    encoders: Dict[str, LabelEncoder] = {}

    for c in feat_cols:
        if df[c].dtype == object or c in CAT_COLS:
            le = LabelEncoder()
            X[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return X, y, encoders, feat_cols


# Cache kết quả huấn luyện
_ml_state: Dict[str, Any] = {
    "trained": False,
    "metrics": {},
    "models": {},
    "scaler": None,
    "encoders": {},
    "feature_names": [],
    "X_test": None,
    "y_test": None,
    "y_pred": {},
    "y_prob": {},
    "error": None,
}


def _train_all_models(force: bool = False) -> Dict[str, Any]:
    if _ml_state["trained"] and not force:
        return _ml_state
    if _df is None or len(_df) < 50:
        _ml_state["error"] = "Dataset chưa sẵn sàng hoặc quá nhỏ."
        return _ml_state

    try:
        X, y, encoders, feat_names = _prepare_xy(_df)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        }

        metrics: Dict[str, Any] = {}
        y_pred_map: Dict[str, Any] = {}
        y_prob_map: Dict[str, Any] = {}

        for name, mdl in candidates.items():
            if name == "Logistic Regression":
                mdl.fit(X_tr_s, y_tr)
                y_pred = mdl.predict(X_te_s)
                y_prob = mdl.predict_proba(X_te_s)[:, 1]
            else:
                mdl.fit(X_tr, y_tr)
                y_pred = mdl.predict(X_te)
                y_prob = mdl.predict_proba(X_te)[:, 1]

            cm = confusion_matrix(y_te, y_pred).tolist()
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            roc_auc = auc(fpr, tpr)

            metrics[name] = {
                "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
                "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
                "auc": round(float(roc_auc), 4),
                "confusion_matrix": cm,
                "roc": {
                    "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//40)]],
                    "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//40)]],
                },
            }
            y_pred_map[name] = y_pred.tolist()
            y_prob_map[name] = y_prob.tolist()

        # Feature importance cho Random Forest
        rf = candidates["Random Forest"]
        importances = sorted(
            [{"feature": f, "importance": round(float(v), 4)}
             for f, v in zip(feat_names, rf.feature_importances_)],
            key=lambda x: -x["importance"],
        )

        _ml_state.update({
            "trained": True,
            "metrics": metrics,
            "models": candidates,
            "scaler": scaler,
            "encoders": encoders,
            "feature_names": feat_names,
            "X_test": X_te,
            "y_test": y_te.tolist(),
            "y_pred": y_pred_map,
            "y_prob": y_prob_map,
            "feature_importance": importances,
            "class_balance": {
                "alive": int((y == 0).sum()),
                "dead": int((y == 1).sum()),
            },
            "error": None,
        })
    except Exception as exc:
        _ml_state["error"] = str(exc)
        _ml_state["trained"] = False

    return _ml_state


# ==============================================================
# 2. SCHEMAS  —  30 đặc trưng load_breast_cancer (gốc)
# ==============================================================
class BreastCancerInput(BaseModel):
    mean_radius: float = Field(..., ge=0)
    mean_texture: float = Field(..., ge=0)
    mean_perimeter: float = Field(..., ge=0)
    mean_area: float = Field(..., ge=0)
    mean_smoothness: float = Field(..., ge=0)
    mean_compactness: float = Field(..., ge=0)
    mean_concavity: float = Field(..., ge=0)
    mean_concave_points: float = Field(..., ge=0)
    mean_symmetry: float = Field(..., ge=0)
    mean_fractal_dimension: float = Field(..., ge=0)
    radius_error: float = Field(..., ge=0)
    texture_error: float = Field(..., ge=0)
    perimeter_error: float = Field(..., ge=0)
    area_error: float = Field(..., ge=0)
    smoothness_error: float = Field(..., ge=0)
    compactness_error: float = Field(..., ge=0)
    concavity_error: float = Field(..., ge=0)
    concave_points_error: float = Field(..., ge=0)
    symmetry_error: float = Field(..., ge=0)
    fractal_dimension_error: float = Field(..., ge=0)
    worst_radius: float = Field(..., ge=0)
    worst_texture: float = Field(..., ge=0)
    worst_perimeter: float = Field(..., ge=0)
    worst_area: float = Field(..., ge=0)
    worst_smoothness: float = Field(..., ge=0)
    worst_compactness: float = Field(..., ge=0)
    worst_concavity: float = Field(..., ge=0)
    worst_concave_points: float = Field(..., ge=0)
    worst_symmetry: float = Field(..., ge=0)
    worst_fractal_dimension: float = Field(..., ge=0)


class FilterPayload(BaseModel):
    filters: Optional[Dict[str, Any]] = None
    page: int = 1
    page_size: int = 15
    sort_by: Optional[str] = None
    sort_dir: str = "asc"


class PatientInput(BaseModel):
    age: int
    race: str
    marital_status: str
    t_stage: str
    n_stage: str
    stage_6th: str
    differentiate: str
    grade: int
    a_stage: str
    tumor_size: int
    estrogen_status: str
    progesterone_status: str
    regional_node_examined: int
    regional_node_positive: int
    survival_months: int
    model: str = "Random Forest"


CANCER_CLASSES = {0: "malignant", 1: "benign"}
CLASS_LABELS = ["malignant", "benign"]


# ==============================================================
# 3. FASTAPI APP
# ==============================================================
app = FastAPI(
    title="Breast Cancer SVM Diagnostic API",
    description="SVM classifier cho bộ dữ liệu Breast Cancer (30 đặc trưng) + dashboard phân tích sống còn.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "error": _model_error,
        "dataset_loaded": _df is not None,
        "dataset_rows": int(len(_df)) if _df is not None else 0,
        "dataset_error": _df_error,
    }


# ---------------- SVM predict (gốc) ----------------
@app.post("/predict")
def predict(data: BreastCancerInput):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Mô hình chưa sẵn sàng. {_model_error or ''}".strip(),
        )

    features: List[float] = [
        data.mean_radius, data.mean_texture, data.mean_perimeter, data.mean_area,
        data.mean_smoothness, data.mean_compactness, data.mean_concavity,
        data.mean_concave_points, data.mean_symmetry, data.mean_fractal_dimension,
        data.radius_error, data.texture_error, data.perimeter_error, data.area_error,
        data.smoothness_error, data.compactness_error, data.concavity_error,
        data.concave_points_error, data.symmetry_error, data.fractal_dimension_error,
        data.worst_radius, data.worst_texture, data.worst_perimeter, data.worst_area,
        data.worst_smoothness, data.worst_compactness, data.worst_concavity,
        data.worst_concave_points, data.worst_symmetry, data.worst_fractal_dimension,
    ]

    X = np.asarray([features], dtype=float)
    try:
        pred = int(_model.predict(X)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {exc}")

    probabilities = None
    try:
        if hasattr(_model, "predict_proba"):
            raw = _model.predict_proba(X)[0]
            probabilities = [round(float(p), 4) for p in raw]
    except Exception:
        probabilities = None

    return {
        "class_id": pred,
        "prediction": CANCER_CLASSES.get(pred, "unknown"),
        "probabilities": probabilities,
        "labels": CLASS_LABELS,
    }


# ==============================================================
# 3b. DASHBOARD APIs
# ==============================================================
@app.get("/api/meta")
def api_meta():
    """Danh sách cột + giá trị category để FE dựng filter."""
    if _df is None:
        return {"columns": [], "categories": {}, "numeric_ranges": {}}

    categories = {}
    for c in CAT_COLS + ["status"]:
        if c in _df.columns:
            categories[c] = sorted(_df[c].dropna().astype(str).unique().tolist())

    numeric_ranges = {}
    for c in NUM_COLS:
        if c in _df.columns:
            numeric_ranges[c] = {
                "min": _safe(_df[c].min()),
                "max": _safe(_df[c].max()),
            }

    return {
        "columns": _df.columns.tolist(),
        "categories": categories,
        "numeric_ranges": numeric_ranges,
        "rows": int(len(_df)),
        "error": _df_error,
    }


# ---------- 1. Tổng quan dữ liệu ----------
@app.get("/api/overview")
def api_overview():
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    df = _df
    target_dist = (
        df["status"].value_counts().to_dict() if "status" in df.columns else {}
    )
    missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "missing": missing,
        "target_distribution": {k: int(v) for k, v in target_dist.items()},
        "head": [
            {k: _safe(v) for k, v in row.items()}
            for row in df.head(8).to_dict(orient="records")
        ],
        "numeric_summary": {
            c: {
                "mean": _safe(df[c].mean()),
                "std": _safe(df[c].std()),
                "min": _safe(df[c].min()),
                "max": _safe(df[c].max()),
                "median": _safe(df[c].median()),
            }
            for c in NUM_COLS if c in df.columns
        },
    }


# ---------- 2. Lọc + Bảng dữ liệu ----------
@app.post("/api/data")
def api_data(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    # sort
    if payload.sort_by and payload.sort_by in d.columns:
        d = d.sort_values(
            payload.sort_by,
            ascending=(payload.sort_dir.lower() != "desc"),
            kind="mergesort",
        )

    total = int(len(d))
    page = max(1, int(payload.page))
    size = max(1, min(200, int(payload.page_size)))
    start = (page - 1) * size
    end = start + size

    rows = [
        {k: _safe(v) for k, v in r.items()}
        for r in d.iloc[start:end].to_dict(orient="records")
    ]

    return {
        "total": total,
        "page": page,
        "page_size": size,
        "pages": max(1, math.ceil(total / size)),
        "rows": rows,
        "columns": d.columns.tolist(),
    }


# ---------- 3. Đặc điểm bệnh nhân ----------
@app.post("/api/analysis/patients")
def api_patients(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    age_bins = [0, 30, 40, 50, 60, 70, 80, 200]
    age_labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    age_hist = (
        pd.cut(d["age"], bins=age_bins, labels=age_labels, right=False)
        .value_counts().reindex(age_labels).fillna(0).astype(int).to_dict()
    )

    def _dist(col):
        return {str(k): int(v) for k, v in d[col].value_counts().to_dict().items()}

    # race vs survival status
    race_status = {}
    if "status" in d.columns:
        ct = pd.crosstab(d["race"], d["status"])
        race_status = {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                       for k, v in ct.to_dict(orient="index").items()}

    marital_status = {}
    if "status" in d.columns:
        ct = pd.crosstab(d["marital_status"], d["status"])
        marital_status = {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                          for k, v in ct.to_dict(orient="index").items()}

    return {
        "total": int(len(d)),
        "age_hist": age_hist,
        "age_stats": {
            "mean": _safe(d["age"].mean()),
            "median": _safe(d["age"].median()),
            "min": _safe(d["age"].min()),
            "max": _safe(d["age"].max()),
        },
        "race_dist": _dist("race"),
        "marital_dist": _dist("marital_status"),
        "race_by_status": race_status,
        "marital_by_status": marital_status,
    }


# ---------- 4. Đặc điểm bệnh & khối u ----------
@app.post("/api/analysis/disease")
def api_disease(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    size_bins = [0, 10, 20, 30, 40, 50, 75, 100, 1000]
    size_labels = ["<10", "10-19", "20-29", "30-39", "40-49", "50-74", "75-99", "100+"]
    size_hist = (
        pd.cut(d["tumor_size"], bins=size_bins, labels=size_labels, right=False)
        .value_counts().reindex(size_labels).fillna(0).astype(int).to_dict()
    )

    def _dist(col):
        return {str(k): int(v) for k, v in d[col].value_counts().to_dict().items()}

    avg_size_by_stage = (
        d.groupby("stage_6th")["tumor_size"].mean().round(2).to_dict()
        if "stage_6th" in d.columns else {}
    )
    avg_size_by_grade = (
        d.groupby("grade")["tumor_size"].mean().round(2).to_dict()
        if "grade" in d.columns else {}
    )

    return {
        "total": int(len(d)),
        "size_hist": size_hist,
        "size_stats": {
            "mean": _safe(d["tumor_size"].mean()),
            "median": _safe(d["tumor_size"].median()),
            "min": _safe(d["tumor_size"].min()),
            "max": _safe(d["tumor_size"].max()),
        },
        "stage_dist": _dist("stage_6th"),
        "grade_dist": _dist("grade"),
        "differentiate_dist": _dist("differentiate"),
        "t_stage_dist": _dist("t_stage"),
        "n_stage_dist": _dist("n_stage"),
        "a_stage_dist": _dist("a_stage"),
        "avg_size_by_stage": {str(k): float(v) for k, v in avg_size_by_stage.items()},
        "avg_size_by_grade": {str(k): float(v) for k, v in avg_size_by_grade.items()},
    }


# ---------- 5. Yếu tố sinh học & hạch ----------
@app.post("/api/analysis/biology")
def api_biology(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    def _dist(col):
        return {str(k): int(v) for k, v in d[col].value_counts().to_dict().items()}

    # Estrogen x Progesterone crosstab
    ep_crosstab = {}
    if "estrogen_status" in d.columns and "progesterone_status" in d.columns:
        ct = pd.crosstab(d["estrogen_status"], d["progesterone_status"])
        ep_crosstab = {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                       for k, v in ct.to_dict(orient="index").items()}

    # Lymph node positive hist
    node_bins = [-1, 0, 1, 3, 5, 10, 100]
    node_labels = ["0", "1", "2-3", "4-5", "6-10", "10+"]
    node_hist = (
        pd.cut(d["regional_node_positive"], bins=node_bins, labels=node_labels)
        .value_counts().reindex(node_labels).fillna(0).astype(int).to_dict()
    )

    # avg nodes positive by stage
    avg_nodes_by_stage = (
        d.groupby("stage_6th")["regional_node_positive"].mean().round(2).to_dict()
        if "stage_6th" in d.columns else {}
    )

    # survival by hormone status
    surv_by_est = (
        d.groupby("estrogen_status")["survival_months"].mean().round(2).to_dict()
        if "estrogen_status" in d.columns else {}
    )
    surv_by_prog = (
        d.groupby("progesterone_status")["survival_months"].mean().round(2).to_dict()
        if "progesterone_status" in d.columns else {}
    )

    return {
        "total": int(len(d)),
        "estrogen_dist": _dist("estrogen_status"),
        "progesterone_dist": _dist("progesterone_status"),
        "ep_crosstab": ep_crosstab,
        "node_hist": node_hist,
        "node_stats": {
            "mean_examined": _safe(d["regional_node_examined"].mean()),
            "mean_positive": _safe(d["regional_node_positive"].mean()),
            "max_positive": _safe(d["regional_node_positive"].max()),
        },
        "avg_nodes_by_stage": {str(k): float(v) for k, v in avg_nodes_by_stage.items()},
        "survival_by_estrogen": {str(k): float(v) for k, v in surv_by_est.items()},
        "survival_by_progesterone": {str(k): float(v) for k, v in surv_by_prog.items()},
    }


# ---------- 6. Sống còn & thống kê ----------
@app.post("/api/analysis/survival")
def api_survival(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    # histogram survival months
    bins = list(range(0, 121, 10))
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
    surv_hist = (
        pd.cut(d["survival_months"], bins=bins, labels=labels, right=False)
        .value_counts().reindex(labels).fillna(0).astype(int).to_dict()
    )

    # correlation matrix (numeric)
    numeric_df = d.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(3).fillna(0).to_dict() if len(numeric_df.columns) else {}

    # descriptive stats
    desc = {}
    for c in numeric_df.columns:
        s = numeric_df[c]
        desc[c] = {
            "count": int(s.count()),
            "mean": _safe(s.mean()),
            "std": _safe(s.std()),
            "min": _safe(s.min()),
            "q25": _safe(s.quantile(0.25)),
            "median": _safe(s.median()),
            "q75": _safe(s.quantile(0.75)),
            "max": _safe(s.max()),
        }

    # survival by stage
    by_stage = (
        d.groupby("stage_6th")["survival_months"].mean().round(2).to_dict()
        if "stage_6th" in d.columns else {}
    )
    by_grade = (
        d.groupby("grade")["survival_months"].mean().round(2).to_dict()
        if "grade" in d.columns else {}
    )
    by_status = (
        d.groupby("status")["survival_months"].mean().round(2).to_dict()
        if "status" in d.columns else {}
    )

    return {
        "total": int(len(d)),
        "survival_hist": surv_hist,
        "survival_stats": {
            "mean": _safe(d["survival_months"].mean()),
            "median": _safe(d["survival_months"].median()),
            "min": _safe(d["survival_months"].min()),
            "max": _safe(d["survival_months"].max()),
        },
        "correlation": corr,
        "correlation_columns": numeric_df.columns.tolist(),
        "descriptive": desc,
        "survival_by_stage": {str(k): float(v) for k, v in by_stage.items()},
        "survival_by_grade": {str(k): float(v) for k, v in by_grade.items()},
        "survival_by_status": {str(k): float(v) for k, v in by_status.items()},
    }


# ---------- 7. ML ----------
@app.post("/api/ml/train")
def api_ml_train(force: bool = Query(False)):
    state = _train_all_models(force=force)
    if state.get("error"):
        raise HTTPException(status_code=500, detail=state["error"])
    return {
        "trained": state["trained"],
        "metrics": state["metrics"],
        "class_balance": state.get("class_balance", {}),
        "feature_importance": state.get("feature_importance", []),
        "feature_names": state.get("feature_names", []),
    }


@app.get("/api/ml/metrics")
def api_ml_metrics():
    if not _ml_state["trained"]:
        _train_all_models()
    if _ml_state.get("error"):
        raise HTTPException(status_code=500, detail=_ml_state["error"])
    return {
        "metrics": _ml_state["metrics"],
        "class_balance": _ml_state.get("class_balance", {}),
        "feature_importance": _ml_state.get("feature_importance", []),
        "feature_names": _ml_state.get("feature_names", []),
    }


@app.post("/api/ml/predict")
def api_ml_predict(p: PatientInput):
    if not _ml_state["trained"]:
        _train_all_models()
    if _ml_state.get("error"):
        raise HTTPException(status_code=500, detail=_ml_state["error"])

    model_name = p.model
    if model_name not in _ml_state["models"]:
        model_name = "Random Forest"

    # Tạo DataFrame 1 dòng
    row = {
        "age": p.age, "race": p.race, "marital_status": p.marital_status,
        "t_stage": p.t_stage, "n_stage": p.n_stage, "stage_6th": p.stage_6th,
        "differentiate": p.differentiate, "grade": p.grade, "a_stage": p.a_stage,
        "tumor_size": p.tumor_size,
        "estrogen_status": p.estrogen_status,
        "progesterone_status": p.progesterone_status,
        "regional_node_examined": p.regional_node_examined,
        "regional_node_positive": p.regional_node_positive,
        "survival_months": p.survival_months,
    }

    X_row = pd.DataFrame([row])
    encoders = _ml_state["encoders"]
    feat_names = _ml_state["feature_names"]

    for c in feat_names:
        if c in encoders:
            try:
                X_row[c] = encoders[c].transform(X_row[c].astype(str))
            except Exception:
                # nhãn chưa từng thấy → dùng class đầu tiên
                X_row[c] = 0
        else:
            X_row[c] = pd.to_numeric(X_row[c], errors="coerce").fillna(0.0)

    X_row = X_row[feat_names]
    mdl = _ml_state["models"][model_name]

    if model_name == "Logistic Regression":
        X_in = _ml_state["scaler"].transform(X_row)
    else:
        X_in = X_row

    pred = int(mdl.predict(X_in)[0])
    prob = float(mdl.predict_proba(X_in)[0][1])

    return {
        "model": model_name,
        "prediction": "Dead" if pred == 1 else "Alive",
        "class_id": pred,
        "probability_dead": round(prob, 4),
        "probability_alive": round(1.0 - prob, 4),
    }


@app.get("/api/ml/compare")
def api_ml_compare():
    if not _ml_state["trained"]:
        _train_all_models()
    if _ml_state.get("error"):
        raise HTTPException(status_code=500, detail=_ml_state["error"])

    rows = []
    for name, m in _ml_state["metrics"].items():
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "auc": m["auc"],
        })
    rows.sort(key=lambda r: -r["auc"])
    return {"comparison": rows, "best": rows[0]["model"] if rows else None}


# ==============================================================
# 4. FRONTEND
# ==============================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_TEMPLATE


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi" data-theme="light">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Breast Cancer · Dashboard phân tích & SVM</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg-1:#eef2f8;--bg-2:#e0e7ff;--bg-3:#f0f9ff;
  --surface:#fff;--surface-2:#f8fafc;--surface-3:#f1f5f9;
  --text:#0f172a;--text-muted:#64748b;--text-soft:#94a3b8;
  --border:#e2e8f0;--border-strong:#cbd5e1;
  --primary:#2563eb;--primary-dark:#1d4ed8;--primary-soft:#dbeafe;--primary-glow:rgba(37,99,235,.25);
  --violet:#7c3aed;--violet-soft:#ede9fe;
  --danger:#dc2626;--danger-dark:#b91c1c;--danger-soft:#fef2f2;--danger-border:#fecaca;
  --success:#16a34a;--success-dark:#15803d;--success-soft:#f0fdf4;--success-border:#bbf7d0;
  --warn:#d97706;--warn-soft:#fffbeb;
  --radius-sm:8px;--radius:12px;--radius-lg:16px;--radius-xl:22px;
  --shadow-xs:0 1px 2px rgba(15,23,42,.05);
  --shadow-sm:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04);
  --shadow:0 4px 12px rgba(15,23,42,.08),0 2px 4px rgba(15,23,42,.04);
  --shadow-lg:0 20px 40px -16px rgba(15,23,42,.18),0 8px 16px -8px rgba(15,23,42,.08);
  --font:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --transition:180ms cubic-bezier(0.4,0,0.2,1);
}
[data-theme="dark"]{
  --bg-1:#020617;--bg-2:#0b132b;--bg-3:#1e1b4b;
  --surface:#0f172a;--surface-2:#111c33;--surface-3:#1e293b;
  --text:#f1f5f9;--text-muted:#94a3b8;--text-soft:#64748b;
  --border:#1e293b;--border-strong:#334155;
  --primary:#60a5fa;--primary-dark:#3b82f6;--primary-soft:rgba(96,165,250,.14);--primary-glow:rgba(96,165,250,.3);
  --violet:#a78bfa;--violet-soft:rgba(167,139,250,.14);
  --danger:#f87171;--danger-dark:#ef4444;--danger-soft:rgba(248,113,113,.1);--danger-border:rgba(248,113,113,.3);
  --success:#4ade80;--success-dark:#22c55e;--success-soft:rgba(74,222,128,.1);--success-border:rgba(74,222,128,.3);
  --warn:#fbbf24;--warn-soft:rgba(251,191,36,.1);
}
*,*::before,*::after{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font-family:var(--font);font-size:14px;line-height:1.55;color:var(--text);
  background:radial-gradient(1200px 620px at 6% -10%,var(--bg-2) 0,transparent 60%),
             radial-gradient(1000px 520px at 100% 0,var(--bg-3) 0,transparent 55%),var(--bg-1);
  background-attachment:fixed;padding:22px 16px 60px;-webkit-font-smoothing:antialiased;}
h1,h2,h3,p{margin:0}
button,input,select{font:inherit;color:inherit}
button{cursor:pointer}
:focus-visible{outline:2px solid var(--primary);outline-offset:2px;border-radius:4px}

.app{max-width:1280px;margin:0 auto;display:flex;flex-direction:column;gap:18px}

/* header */
.header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);
  padding:16px 22px;box-shadow:var(--shadow);position:relative;overflow:hidden}
.header::before{content:"";position:absolute;inset:0;
  background:linear-gradient(120deg,transparent 40%,var(--primary-soft) 100%);opacity:.55;pointer-events:none}
.brand{display:flex;align-items:center;gap:14px;position:relative;z-index:1}
.brand__icon{width:48px;height:48px;display:grid;place-items:center;border-radius:14px;
  background:linear-gradient(135deg,var(--primary),var(--violet));color:#fff;font-size:22px;
  box-shadow:0 10px 20px -6px var(--primary-glow)}
.brand__title{font-size:19px;font-weight:800;letter-spacing:-.3px}
.brand__sub{font-size:12.5px;color:var(--text-muted);margin-top:2px}
.header__actions{display:flex;align-items:center;gap:10px;position:relative;z-index:1}

.status{font-size:12px;font-weight:600;padding:7px 13px;border-radius:999px;
  background:var(--surface-3);color:var(--text-muted);border:1px solid var(--border);
  display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.status::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 6px currentColor}
.status.is-online{background:var(--success-soft);color:var(--success);border-color:var(--success-border)}
.status.is-offline{background:var(--danger-soft);color:var(--danger);border-color:var(--danger-border)}
.status.is-checking{background:var(--warn-soft);color:var(--warn);border-color:#fde68a}

.theme-btn{width:38px;height:38px;display:grid;place-items:center;background:var(--surface-3);
  border:1px solid var(--border);border-radius:10px;font-size:16px;transition:all var(--transition)}
.theme-btn:hover{background:var(--primary-soft);border-color:var(--primary);transform:translateY(-1px)}

/* tabs */
.tabs{display:flex;gap:4px;flex-wrap:wrap;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:6px;box-shadow:var(--shadow-sm)}
.tab{display:inline-flex;align-items:center;gap:6px;padding:9px 14px;font-size:12.5px;font-weight:600;
  border-radius:var(--radius);border:none;background:transparent;color:var(--text-muted);transition:all var(--transition);white-space:nowrap}
.tab:hover{background:var(--surface-3);color:var(--text)}
.tab.is-active{background:linear-gradient(135deg,var(--primary),var(--violet));color:#fff;
  box-shadow:0 6px 14px -6px var(--primary-glow)}
@media (max-width:640px){.tab{font-size:11.5px;padding:7px 10px}}

/* panel */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);
  box-shadow:var(--shadow);overflow:hidden}
.panel__head{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:14px 20px;border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,var(--surface),var(--surface-2));flex-wrap:wrap}
.panel__head h2{font-size:14.5px;font-weight:700;display:flex;align-items:center;gap:8px}
.panel__head h2::before{content:"";width:4px;height:16px;border-radius:2px;
  background:linear-gradient(180deg,var(--primary),var(--violet))}
.panel__body{padding:18px 20px}
.tabpanel{display:none;flex-direction:column;gap:18px}
.tabpanel.is-active{display:flex}

/* stat cards */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat{padding:14px 16px;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--surface-2);transition:all var(--transition)}
.stat:hover{border-color:var(--primary);transform:translateY(-1px);box-shadow:var(--shadow-sm)}
.stat__label{font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em}
.stat__value{font-size:22px;font-weight:800;font-family:var(--mono);margin-top:4px;letter-spacing:-.5px}
.stat__value.primary{color:var(--primary)}
.stat__value.danger{color:var(--danger)}
.stat__value.success{color:var(--success)}
.stat__value.violet{color:var(--violet)}

/* chart cards */
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.chart-card{border:1px solid var(--border);border-radius:var(--radius);padding:14px;
  background:var(--surface-2);display:flex;flex-direction:column;gap:8px}
.chart-card__title{font-size:12px;font-weight:700;color:var(--text);letter-spacing:.02em}
.chart-wrap{position:relative;height:260px}
.chart-wrap.tall{height:340px}

/* table */
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface)}
table.dt{width:100%;border-collapse:collapse;font-size:12.5px;min-width:800px}
table.dt th,table.dt td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
table.dt th{background:var(--surface-3);font-weight:700;font-size:11px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--text-muted);position:sticky;top:0;cursor:pointer}
table.dt th:hover{color:var(--primary)}
table.dt tbody tr:hover{background:var(--surface-2)}
table.dt td.mono{font-family:var(--mono)}
.pagination{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:12px;flex-wrap:wrap}
.pagination__info{font-size:12px;color:var(--text-muted)}
.pagination__btns{display:flex;gap:6px}
.pg-btn{padding:6px 12px;font-size:12px;font-weight:600;border-radius:8px;
  border:1px solid var(--border);background:var(--surface);color:var(--text-muted)}
.pg-btn:hover:not(:disabled){border-color:var(--primary);color:var(--primary)}
.pg-btn:disabled{opacity:.4;cursor:not-allowed}

/* filter */
.filter-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.filter-field label{display:block;font-size:11px;font-weight:600;color:var(--text-muted);
  margin-bottom:5px;letter-spacing:.02em}
.filter-field input,.filter-field select{width:100%;padding:8px 10px;font-size:12.5px;
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);color:var(--text)}
.filter-field input:focus,.filter-field select:focus{outline:none;border-color:var(--primary);
  box-shadow:0 0 0 3px var(--primary-glow)}
.range-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.chip-group{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.chip{padding:5px 10px;font-size:11.5px;font-weight:600;border-radius:999px;
  border:1px solid var(--border);background:var(--surface);color:var(--text-muted);transition:all var(--transition)}
.chip.is-on{background:var(--primary-soft);color:var(--primary);border-color:var(--primary)}

/* buttons (gốc) */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:10px 20px;
  font-size:13.5px;font-weight:600;border-radius:var(--radius);border:1px solid transparent;transition:all var(--transition)}
.btn:active:not(:disabled){transform:translateY(1px)}
.btn:disabled{opacity:.55;cursor:not-allowed}
.btn--primary{background:linear-gradient(135deg,var(--primary),var(--violet));color:#fff;
  box-shadow:0 8px 20px -8px var(--primary-glow);padding-inline:28px}
.btn--primary:hover:not(:disabled){box-shadow:0 12px 26px -8px var(--primary-glow);transform:translateY(-1px)}
.btn--ghost{background:var(--surface);color:var(--text-muted);border-color:var(--border)}
.btn--ghost:hover:not(:disabled){background:var(--surface-3);color:var(--text);border-color:var(--border-strong)}
.btn--danger-ghost{background:var(--danger-soft);color:var(--danger);border-color:var(--danger-border)}
.btn--danger-ghost:hover:not(:disabled){background:var(--danger);color:#fff}
.btn--success-ghost{background:var(--success-soft);color:var(--success);border-color:var(--success-border)}
.btn--success-ghost:hover:not(:disabled){background:var(--success);color:#fff}
.btn--sm{padding:7px 14px;font-size:12px}

.spinner{width:14px;height:14px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;
  border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* form (SVM gốc) */
.groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;padding:18px 20px}
.group{border:1px solid var(--border);border-radius:var(--radius);padding:14px;background:var(--surface-2)}
.group__title{font-size:11.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
  color:var(--primary);margin-bottom:12px;display:flex;align-items:center;gap:8px}
.group__title::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--primary);
  box-shadow:0 0 0 3px var(--primary-soft)}
.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(126px,1fr));gap:10px}
.field label{display:block;font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:5px}
.field input{width:100%;padding:9px 11px;font-size:13px;font-family:var(--mono);
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);color:var(--text)}
.field input:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px var(--primary-glow)}
.field input.is-invalid{border-color:var(--danger);background:var(--danger-soft)}
.actions{display:flex;gap:10px;flex-wrap:wrap;padding:16px 20px;border-top:1px solid var(--border);background:var(--surface-2)}

.result__empty{padding:40px 20px;text-align:center;color:var(--text-muted)}
.result__empty .icon{font-size:40px;opacity:.3;display:block;margin-bottom:12px}
.result__card{padding:20px;animation:fadeUp .35s cubic-bezier(.2,.9,.3,1)}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.verdict{border-radius:var(--radius-lg);padding:24px 16px;text-align:center;margin-bottom:18px}
.verdict--malignant{background:var(--danger-soft);border:1px solid var(--danger-border)}
.verdict--benign{background:var(--success-soft);border:1px solid var(--success-border)}
.verdict__icon{font-size:40px;line-height:1;margin-bottom:10px}
.verdict__label{font-size:20px;font-weight:800;letter-spacing:.04em;text-transform:uppercase}
.verdict--malignant .verdict__label{color:var(--danger)}
.verdict--benign .verdict__label{color:var(--success)}
.verdict__sub{font-size:12.5px;color:var(--text-muted);margin-top:4px}
.prob{margin-bottom:16px}
.prob__head{display:flex;justify-content:space-between;font-size:12px;font-weight:600;margin-bottom:6px}
.prob__track{height:8px;border-radius:999px;background:var(--surface-3);overflow:hidden}
.prob__fill{height:100%;border-radius:999px;transition:width .55s cubic-bezier(.2,.9,.3,1)}
.prob__fill--danger{background:linear-gradient(90deg,#f87171,#dc2626)}
.prob__fill--success{background:linear-gradient(90deg,#4ade80,#16a34a)}
.meta__row{display:flex;justify-content:space-between;gap:12px;padding:10px 0;font-size:13px;
  border-bottom:1px dashed var(--border)}
.meta__row:last-child{border-bottom:none}
.meta__key{color:var(--text-muted)}
.meta__val{font-weight:600;font-family:var(--mono)}
.alert{margin:0 20px 18px;padding:14px 16px;border-radius:var(--radius);
  background:var(--danger-soft);border:1px solid var(--danger-border);color:var(--danger-dark);
  font-size:13px;line-height:1.6}
.note{margin-top:14px;padding:11px 14px;font-size:11.5px;line-height:1.65;color:var(--text-muted);
  background:var(--surface-2);border-left:3px solid var(--border-strong);border-radius:0 8px 8px 0}

.footer{text-align:center;font-size:12px;color:var(--text-muted);padding-top:6px}
.muted{color:var(--text-muted);font-size:12.5px}
</style>
</head>

<body>
<div class="app">

  <header class="header">
    <div class="brand">
      <div class="brand__icon" aria-hidden="true">⚕</div>
      <div>
        <h1 class="brand__title">Breast Cancer Analytics</h1>
        <p class="brand__sub">Dashboard phân tích sống còn + SVM chẩn đoán 30 đặc trưng</p>
      </div>
    </div>
    <div class="header__actions">
      <span class="status" id="status">Đang kiểm tra…</span>
      <button class="theme-btn" id="btnTheme" type="button" title="Đổi sáng/tối">🌙</button>
    </div>
  </header>

  <nav class="tabs" id="tabs">
    <button class="tab is-active" data-tab="overview">📌 Tổng quan</button>
    <button class="tab" data-tab="data">🔎 Lọc &amp; dữ liệu</button>
    <button class="tab" data-tab="patients">📊 Bệnh nhân</button>
    <button class="tab" data-tab="disease">🩺 Bệnh &amp; khối u</button>
    <button class="tab" data-tab="biology">🧬 Sinh học &amp; hạch</button>
    <button class="tab" data-tab="survival">⏱️ Sống còn &amp; thống kê</button>
    <button class="tab" data-tab="ml">🤖 ML &amp; dự đoán</button>
    <button class="tab" data-tab="svm">🧪 SVM 30 đặc trưng</button>
  </nav>

  <!-- 1. TỔNG QUAN -->
  <section class="tabpanel is-active" data-panel="overview">
    <div class="panel">
      <div class="panel__head">
        <h2>Tổng quan dữ liệu</h2>
        <button class="btn btn--ghost btn--sm" id="btnRefreshOverview">🔄 Làm mới</button>
      </div>
      <div class="panel__body">
        <div class="stat-grid" id="ovStats"></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel__head"><h2>Phân bố biến mục tiêu</h2></div>
      <div class="panel__body">
        <div class="chart-grid">
          <div class="chart-card">
            <div class="chart-card__title">Status (Alive / Dead)</div>
            <div class="chart-wrap"><canvas id="ovTarget"></canvas></div>
          </div>
          <div class="chart-card">
            <div class="chart-card__title">Kiểu dữ liệu các cột</div>
            <div class="chart-wrap"><canvas id="ovDtypes"></canvas></div>
          </div>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="panel__head"><h2>8 dòng đầu tiên</h2></div>
      <div class="panel__body">
        <div class="table-wrap" id="ovHead"></div>
      </div>
    </div>
  </section>

  <!-- 2. LỌC + DỮ LIỆU -->
  <section class="tabpanel" data-panel="data">
    <div class="panel">
      <div class="panel__head">
        <h2>Bộ lọc</h2>
        <div style="display:flex;gap:6px">
          <button class="btn btn--ghost btn--sm" id="btnClearFilter">🧹 Xoá lọc</button>
          <button class="btn btn--primary btn--sm" id="btnApplyFilter">🔎 Áp dụng</button>
        </div>
      </div>
      <div class="panel__body">
        <div class="filter-grid" id="filterGrid"></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel__head">
        <h2>Bảng dữ liệu</h2>
        <span class="muted" id="dataCount">—</span>
      </div>
      <div class="panel__body">
        <div class="table-wrap" id="dataTable"></div>
        <div class="pagination">
          <div class="pagination__info" id="pgInfo">Trang 1</div>
          <div class="pagination__btns">
            <button class="pg-btn" id="pgPrev">‹ Trước</button>
            <button class="pg-btn" id="pgNext">Sau ›</button>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- 3. BỆNH NHÂN -->
  <section class="tabpanel" data-panel="patients">
    <div class="panel">
      <div class="panel__head"><h2>Đặc điểm bệnh nhân</h2>
        <button class="btn btn--ghost btn--sm" data-refresh="patients">🔄</button>
      </div>
      <div class="panel__body"><div class="stat-grid" id="ptStats"></div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div class="chart-card__title">Phân bố độ tuổi</div>
        <div class="chart-wrap"><canvas id="ptAge"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Chủng tộc</div>
        <div class="chart-wrap"><canvas id="ptRace"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Tình trạng hôn nhân</div>
        <div class="chart-wrap"><canvas id="ptMarital"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Race × Status</div>
        <div class="chart-wrap"><canvas id="ptRaceStatus"></canvas></div></div>
    </div>
  </section>

  <!-- 4. BỆNH & KHỐI U -->
  <section class="tabpanel" data-panel="disease">
    <div class="panel">
      <div class="panel__head"><h2>Đặc điểm bệnh &amp; khối u</h2>
        <button class="btn btn--ghost btn--sm" data-refresh="disease">🔄</button>
      </div>
      <div class="panel__body"><div class="stat-grid" id="disStats"></div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div class="chart-card__title">Phân bố kích thước khối u (mm)</div>
        <div class="chart-wrap"><canvas id="disSize"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Giai đoạn 6th Stage</div>
        <div class="chart-wrap"><canvas id="disStage"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Grade</div>
        <div class="chart-wrap"><canvas id="disGrade"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Differentiate</div>
        <div class="chart-wrap"><canvas id="disDiff"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Kích thước TB theo Stage</div>
        <div class="chart-wrap"><canvas id="disSizeStage"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Kích thước TB theo Grade</div>
        <div class="chart-wrap"><canvas id="disSizeGrade"></canvas></div></div>
    </div>
  </section>

  <!-- 5. SINH HỌC & HẠCH -->
  <section class="tabpanel" data-panel="biology">
    <div class="panel">
      <div class="panel__head"><h2>Sinh học &amp; hạch bạch huyết</h2>
        <button class="btn btn--ghost btn--sm" data-refresh="biology">🔄</button>
      </div>
      <div class="panel__body"><div class="stat-grid" id="bioStats"></div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div class="chart-card__title">Estrogen Status</div>
        <div class="chart-wrap"><canvas id="bioEst"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Progesterone Status</div>
        <div class="chart-wrap"><canvas id="bioProg"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Estrogen × Progesterone</div>
        <div class="chart-wrap"><canvas id="bioEP"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Hạch dương tính</div>
        <div class="chart-wrap"><canvas id="bioNode"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Sống còn TB theo Estrogen</div>
        <div class="chart-wrap"><canvas id="bioSurvEst"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Sống còn TB theo Progesterone</div>
        <div class="chart-wrap"><canvas id="bioSurvProg"></canvas></div></div>
    </div>
  </section>

  <!-- 6. SỐNG CÒN -->
  <section class="tabpanel" data-panel="survival">
    <div class="panel">
      <div class="panel__head"><h2>Sống còn &amp; thống kê</h2>
        <button class="btn btn--ghost btn--sm" data-refresh="survival">🔄</button>
      </div>
      <div class="panel__body"><div class="stat-grid" id="svStats"></div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div class="chart-card__title">Phân bố thời gian sống (tháng)</div>
        <div class="chart-wrap"><canvas id="svHist"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Sống còn TB theo Stage</div>
        <div class="chart-wrap"><canvas id="svStage"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Sống còn TB theo Grade</div>
        <div class="chart-wrap"><canvas id="svGrade"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Sống còn TB theo Status</div>
        <div class="chart-wrap"><canvas id="svStatus"></canvas></div></div>
    </div>
    <div class="panel">
      <div class="panel__head"><h2>Ma trận tương quan</h2></div>
      <div class="panel__body">
        <div class="chart-card"><div class="chart-wrap tall"><canvas id="svCorr"></canvas></div></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel__head"><h2>Thống kê mô tả</h2></div>
      <div class="panel__body"><div class="table-wrap" id="svDesc"></div></div>
    </div>
  </section>

  <!-- 7. ML -->
  <section class="tabpanel" data-panel="ml">
    <div class="panel">
      <div class="panel__head">
        <h2>Machine Learning</h2>
        <div style="display:flex;gap:6px">
          <button class="btn btn--ghost btn--sm" id="btnTrain">🏋️ Huấn luyện</button>
          <button class="btn btn--primary btn--sm" id="btnCompare">📊 So sánh mô hình</button>
        </div>
      </div>
      <div class="panel__body">
        <div class="stat-grid" id="mlStats"></div>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div class="chart-card__title">So sánh chỉ số</div>
        <div class="chart-wrap"><canvas id="mlBar"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">ROC Curve</div>
        <div class="chart-wrap"><canvas id="mlRoc"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Feature Importance (Random Forest)</div>
        <div class="chart-wrap tall"><canvas id="mlFeat"></canvas></div></div>
      <div class="chart-card"><div class="chart-card__title">Confusion Matrix (best)</div>
        <div class="chart-wrap"><canvas id="mlCm"></canvas></div></div>
    </div>

    <div class="panel">
      <div class="panel__head"><h2>Dự đoán bệnh nhân mới</h2></div>
      <div class="panel__body">
        <div class="filter-grid" id="mlForm"></div>
        <div style="margin-top:14px;display:flex;gap:8px;flex-wrap:wrap">
          <button class="btn btn--primary" id="btnPredictPatient">🔮 Dự đoán</button>
        </div>
        <div id="mlResult" style="margin-top:14px"></div>
      </div>
    </div>
  </section>

  <!-- 8. SVM gốc -->
  <section class="tabpanel" data-panel="svm">
    <form class="panel" id="form" autocomplete="off" novalidate>
      <div class="panel__head">
        <h2>Chẩn đoán SVM · 30 đặc trưng</h2>
        <div style="display:flex;gap:6px">
          <button type="button" class="btn btn--danger-ghost btn--sm" id="btnSampleMalignant">⚠️ Mẫu ác tính</button>
          <button type="button" class="btn btn--success-ghost btn--sm" id="btnSampleBenign">✅ Mẫu lành tính</button>
        </div>
      </div>
      <div class="groups" id="groups"></div>
      <div class="actions">
        <button type="submit" class="btn btn--primary" id="btnSubmit">
          <span class="spinner" id="spinner" hidden></span>
          <span id="btnText">🔬 Phân tích</span>
        </button>
        <button type="reset" class="btn btn--ghost" id="btnReset">🧹 Xoá tất cả</button>
      </div>
    </form>

    <aside class="panel">
      <div class="panel__head"><h2>Kết quả</h2></div>
      <div class="result__empty" id="resultEmpty">
        <span class="icon">🔬</span>
        <p>Nhập đầy đủ <strong>30 chỉ số</strong> rồi bấm <strong>Phân tích</strong>.</p>
      </div>
      <div id="resultBody" hidden></div>
      <div id="resultError" hidden></div>
    </aside>
  </section>

  <p class="footer">Demo học thuật · Không thay thế chẩn đoán y khoa</p>
</div>


<script>
/* ============================================================
   MODULE 1 · CONFIG (giữ nguyên)
   ============================================================ */
const CONFIG = { API_URL:"/predict", HEALTH_URL:"/health", REQUEST_TIMEOUT_MS:15000 };

const FEATURE_GROUPS = [
  { key:"mean", title:"Giá trị trung bình", fields:[
    ["mean_radius","Bán kính"],["mean_texture","Kết cấu"],["mean_perimeter","Chu vi"],
    ["mean_area","Diện tích"],["mean_smoothness","Độ mịn"],["mean_compactness","Độ đặc"],
    ["mean_concavity","Độ lõm"],["mean_concave_points","Điểm lõm"],["mean_symmetry","Đối xứng"],
    ["mean_fractal_dimension","Fractal"]]},
  { key:"error", title:"Sai số chuẩn", fields:[
    ["radius_error","Bán kính"],["texture_error","Kết cấu"],["perimeter_error","Chu vi"],
    ["area_error","Diện tích"],["smoothness_error","Độ mịn"],["compactness_error","Độ đặc"],
    ["concavity_error","Độ lõm"],["concave_points_error","Điểm lõm"],["symmetry_error","Đối xứng"],
    ["fractal_dimension_error","Fractal"]]},
  { key:"worst", title:"Giá trị xấu nhất", fields:[
    ["worst_radius","Bán kính"],["worst_texture","Kết cấu"],["worst_perimeter","Chu vi"],
    ["worst_area","Diện tích"],["worst_smoothness","Độ mịn"],["worst_compactness","Độ đặc"],
    ["worst_concavity","Độ lõm"],["worst_concave_points","Điểm lõm"],["worst_symmetry","Đối xứng"],
    ["worst_fractal_dimension","Fractal"]]}
];

const SAMPLES = {
  malignant: {mean_radius:17.99,mean_texture:10.38,mean_perimeter:122.8,mean_area:1001.0,
    mean_smoothness:0.1184,mean_compactness:0.2776,mean_concavity:0.3001,mean_concave_points:0.1471,
    mean_symmetry:0.2419,mean_fractal_dimension:0.07871,radius_error:1.095,texture_error:0.9053,
    perimeter_error:8.589,area_error:153.4,smoothness_error:0.006399,compactness_error:0.04904,
    concavity_error:0.05373,concave_points_error:0.01587,symmetry_error:0.03003,
    fractal_dimension_error:0.006193,worst_radius:25.38,worst_texture:17.33,worst_perimeter:184.6,
    worst_area:2019.0,worst_smoothness:0.1622,worst_compactness:0.6656,worst_concavity:0.7119,
    worst_concave_points:0.2654,worst_symmetry:0.4601,worst_fractal_dimension:0.1189},
  benign: {mean_radius:13.54,mean_texture:14.36,mean_perimeter:87.46,mean_area:566.3,
    mean_smoothness:0.09779,mean_compactness:0.08129,mean_concavity:0.06664,mean_concave_points:0.04781,
    mean_symmetry:0.1885,mean_fractal_dimension:0.05766,radius_error:0.2699,texture_error:0.7886,
    perimeter_error:2.058,area_error:23.56,smoothness_error:0.008462,compactness_error:0.0146,
    concavity_error:0.02387,concave_points_error:0.01315,symmetry_error:0.0198,
    fractal_dimension_error:0.0023,worst_radius:15.11,worst_texture:19.26,worst_perimeter:99.7,
    worst_area:711.2,worst_smoothness:0.144,worst_compactness:0.1773,worst_concavity:0.239,
    worst_concave_points:0.1288,worst_symmetry:0.2977,worst_fractal_dimension:0.07259}
};

const LABELS = {0:{en:"Malignant",vi:"Ác tính",icon:"⚠️",cls:"malignant"},
                1:{en:"Benign",vi:"Lành tính",icon:"✅",cls:"benign"}};

/* ============================================================
   MODULE 2 · API
   ============================================================ */
const Api = (() => {
  async function request(url, options={}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch(url, {...options, signal:controller.signal});
      const text = await res.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { data = text; }
      if (!res.ok) {
        const detail = (data && data.detail) ? data.detail : (typeof data === "string" ? data : res.statusText);
        throw new Error(`HTTP ${res.status} — ${detail}`);
      }
      return data;
    } finally { clearTimeout(timer); }
  }
  const post = (url, body) => request(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body||{})});
  return {
    predict:  (payload) => post(CONFIG.API_URL, payload),
    health:   () => request(CONFIG.HEALTH_URL),
    meta:     () => request("/api/meta"),
    overview: () => request("/api/overview"),
    data:     (p) => post("/api/data", p),
    patients: (p) => post("/api/analysis/patients", p),
    disease:  (p) => post("/api/analysis/disease", p),
    biology:  (p) => post("/api/analysis/biology", p),
    survival: (p) => post("/api/analysis/survival", p),
    train:    () => post("/api/ml/train", {}),
    metrics:  () => request("/api/ml/metrics"),
    compare:  () => request("/api/ml/compare"),
    mlPredict:(p) => post("/api/ml/predict", p),
  };
})();

/* ============================================================
   MODULE 3 · CHART REGISTRY (Chart.js helper)
   ============================================================ */
const Charts = (() => {
  const registry = {};
  function palette(i, alpha=0.85) {
    const colors = ["#2563eb","#7c3aed","#dc2626","#16a34a","#d97706","#0891b2","#db2777","#65a30d"];
    const c = colors[i % colors.length];
    return alpha >= 1 ? c : hexA(c, alpha);
  }
  function hexA(hex, a) {
    const h = hex.replace("#",""); const r=parseInt(h.substring(0,2),16),
      g=parseInt(h.substring(2,4),16), b=parseInt(h.substring(4,6),16);
    return `rgba(${r},${g},${b},${a})`;
  }
  const gridColor = () => getComputedStyle(document.documentElement).getPropertyValue("--border").trim() || "#e2e8f0";
  const textColor = () => getComputedStyle(document.documentElement).getPropertyValue("--text-muted").trim() || "#64748b";

  function destroy(id) { if (registry[id]) { registry[id].destroy(); delete registry[id]; } }
  function make(id, cfg) {
    const el = document.getElementById(id);
    if (!el) return null;
    destroy(id);
    registry[id] = new Chart(el.getContext("2d"), cfg);
    return registry[id];
  }
  function baseOpts() {
    return {
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{labels:{color:textColor(), font:{family:"Plus Jakarta Sans", size:11}}} },
      scales:{
        x:{ticks:{color:textColor(), font:{size:10}}, grid:{color:gridColor()}},
        y:{ticks:{color:textColor(), font:{size:10}}, grid:{color:gridColor()}, beginAtZero:true}
      }
    };
  }
  return {make, destroy, baseOpts, palette, hexA, gridColor, textColor};
})();

/* ============================================================
   MODULE 4 · UI (giữ nguyên + mở rộng nhỏ)
   ============================================================ */
const UI = (() => {
  const $ = (id) => document.getElementById(id);

  function buildForm() {
    $("groups").innerHTML = FEATURE_GROUPS.map(g => `
      <section class="group">
        <h3 class="group__title">${g.title}</h3>
        <div class="fields">
          ${g.fields.map(([name,label]) => `
            <div class="field">
              <label for="${name}" title="${name}">${label}</label>
              <input id="${name}" name="${name}" type="number" step="any" min="0" placeholder="0.0" required />
            </div>`).join("")}
        </div>
      </section>`).join("");
  }

  function readForm() {
    const payload = {}; let firstInvalid = null;
    for (const g of FEATURE_GROUPS) for (const [name] of g.fields) {
      const el = $(name); el.classList.remove("is-invalid");
      const raw = el.value.trim();
      if (raw === "") { el.classList.add("is-invalid"); if (!firstInvalid) firstInvalid = {name, reason:"bỏ trống"}; continue; }
      const val = Number(raw);
      if (!Number.isFinite(val)) { el.classList.add("is-invalid"); if (!firstInvalid) firstInvalid = {name, reason:"không phải số"}; continue; }
      if (val < 0) { el.classList.add("is-invalid"); if (!firstInvalid) firstInvalid = {name, reason:"âm"}; continue; }
      payload[name] = val;
    }
    if (firstInvalid) {
      const label = (FEATURE_GROUPS.flatMap(g => g.fields).find(([n]) => n === firstInvalid.name) || [firstInvalid.name, firstInvalid.name])[1];
      throw new Error(`Trường "${label}" (${firstInvalid.name}) ${firstInvalid.reason}.`);
    }
    return payload;
  }

  function fillSample(kind) {
    const data = SAMPLES[kind]; if (!data) return;
    for (const [k,v] of Object.entries(data)) { const el = $(k); if (el) { el.value = v; el.classList.remove("is-invalid"); } }
  }

  function clearResult() {
    $("resultEmpty").hidden = false;
    $("resultBody").hidden = true; $("resultBody").innerHTML = "";
    $("resultError").hidden = true; $("resultError").innerHTML = "";
  }

  function showError(message) {
    $("resultEmpty").hidden = true; $("resultBody").hidden = true;
    const box = $("resultError"); box.hidden = false;
    box.innerHTML = `<div class="alert"><strong>❌ Không thể phân tích</strong>${message}</div>`;
  }

  function showResult(data) {
    const info = LABELS[data.class_id] ?? {en:data.prediction, vi:data.prediction, icon:"•", cls:"benign"};
    const time = new Date().toLocaleTimeString("vi-VN");
    $("resultEmpty").hidden = true; $("resultError").hidden = true;

    let probHTML = "";
    if (Array.isArray(data.probabilities) && data.probabilities.length === 2) {
      const pMal = data.probabilities[0]*100, pBen = data.probabilities[1]*100;
      probHTML = `
        <div class="prob"><div class="prob__head"><span>⚠️ Ác tính</span><span>${pMal.toFixed(2)}%</span></div>
          <div class="prob__track"><div class="prob__fill prob__fill--danger" style="width:${pMal}%"></div></div></div>
        <div class="prob"><div class="prob__head"><span>✅ Lành tính</span><span>${pBen.toFixed(2)}%</span></div>
          <div class="prob__track"><div class="prob__fill prob__fill--success" style="width:${pBen}%"></div></div></div>`;
    }
    const body = $("resultBody"); body.hidden = false;
    body.innerHTML = `
      <div class="result__card">
        <div class="verdict verdict--${info.cls}">
          <div class="verdict__icon">${info.icon}</div>
          <div class="verdict__label">${info.vi}</div>
          <div class="verdict__sub">${info.en}</div>
        </div>
        ${probHTML}
        <div class="meta">
          <div class="meta__row"><span class="meta__key">Mã lớp</span><span class="meta__val">${data.class_id}</span></div>
          <div class="meta__row"><span class="meta__key">Nhãn</span><span class="meta__val">${data.prediction}</span></div>
          <div class="meta__row"><span class="meta__key">Số đặc trưng</span><span class="meta__val">30</span></div>
          <div class="meta__row"><span class="meta__key">Thời điểm</span><span class="meta__val">${time}</span></div>
        </div>
        <div class="note">Kết quả chỉ mang tính tham khảo từ mô hình học máy, không thay thế kết luận của bác sĩ chuyên khoa.</div>
      </div>`;
  }

  function setLoading(on) {
    $("btnSubmit").disabled = on; $("spinner").hidden = !on;
    $("btnText").textContent = on ? "Đang xử lý…" : "🔬 Phân tích";
  }

  function setStatus(state, extra="") {
    const el = $("status"); el.className = "status";
    if (state === "online") { el.classList.add("is-online"); el.textContent = "API sẵn sàng"; }
    else if (state === "offline") { el.classList.add("is-offline"); el.textContent = "Mất kết nối"; }
    else if (state === "checking") { el.classList.add("is-checking"); el.textContent = "Đang kiểm tra…"; }
    else { el.textContent = "Chưa kiểm tra"; }
    if (extra) el.title = extra;
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    $("btnTheme").textContent = theme === "dark" ? "☀️" : "🌙";
    try { localStorage.setItem("bc-theme", theme); } catch {}
  }

  return { buildForm, readForm, fillSample, clearResult, showError, showResult, setLoading, setStatus, applyTheme };
})();

/* ============================================================
   MODULE 5 · TAB NAVIGATION
   ============================================================ */
const Tabs = (() => {
  const tabBtns = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tabpanel");
  const loaded = new Set();

  function show(name) {
    tabBtns.forEach(b => b.classList.toggle("is-active", b.dataset.tab === name));
    panels.forEach(p => p.classList.toggle("is-active", p.dataset.panel === name));
    if (!loaded.has(name)) { loaded.add(name); Dashboard.load(name); }
  }
  function init() { tabBtns.forEach(b => b.addEventListener("click", () => show(b.dataset.tab))); }
  return { init, show, loaded };
})();

/* ============================================================
   MODULE 6 · DASHBOARD (7 nhóm chức năng)
   ============================================================ */
const Dashboard = (() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    meta: null,
    filters: {},
    page: 1,
    pageSize: 15,
    sortBy: null,
    sortDir: "asc",
    ml: null,
  };

  const num = (v, digits=2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" :
    (typeof v === "number" ? v.toFixed(digits) : v);

  function statCard(label, value, cls="") {
    return `<div class="stat"><div class="stat__label">${label}</div>
      <div class="stat__value ${cls}">${value}</div></div>`;
  }

  /* ---------- helpers ---------- */
  function buildTable(containerId, columns, rows) {
    const el = $(containerId); if (!el) return;
    if (!rows || !rows.length) { el.innerHTML = `<div style="padding:20px" class="muted">Không có dữ liệu.</div>`; return; }
    const head = columns.map(c => `<th>${c}</th>`).join("");
    const body = rows.map(r => `<tr>${columns.map(c => {
      const v = r[c]; const isNum = typeof v === "number";
      return `<td class="${isNum?'mono':''}">${v === null || v === undefined ? "" : v}</td>`;
    }).join("")}</tr>`).join("");
    el.innerHTML = `<table class="dt"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  /* ============================================================
     1. TỔNG QUAN
     ============================================================ */
  async function loadOverview() {
    try {
      const d = await Api.overview();
      $("ovStats").innerHTML =
        statCard("Số dòng", d.rows, "primary") +
        statCard("Số cột", d.cols, "violet") +
        statCard("Alive", (d.target_distribution.Alive ?? 0), "success") +
        statCard("Dead", (d.target_distribution.Dead ?? 0), "danger") +
        statCard("Cột thiếu dữ liệu", Object.keys(d.missing).length) +
        statCard("Cột số", Object.values(d.dtypes).filter(t => t.includes("int") || t.includes("float")).length);

      const t = d.target_distribution || {};
      Charts.make("ovTarget", {
        type:"doughnut",
        data:{labels:Object.keys(t), datasets:[{data:Object.values(t),
          backgroundColor:["#16a34a","#dc2626","#7c3aed","#2563eb","#d97706"]}]},
        options:{responsive:true, maintainAspectRatio:false,
          plugins:{legend:{position:"bottom", labels:{color:Charts.textColor()}}}}
      });

      const dtypeCounts = {};
      Object.values(d.dtypes).forEach(t => { dtypeCounts[t] = (dtypeCounts[t]||0)+1; });
      Charts.make("ovDtypes", {
        type:"bar",
        data:{labels:Object.keys(dtypeCounts),
          datasets:[{label:"Số cột", data:Object.values(dtypeCounts), backgroundColor:Charts.palette(0)}]},
        options:Charts.baseOpts()
      });

      const cols = d.columns;
      buildTable("ovHead", cols, d.head);
    } catch(e) { $("ovStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ============================================================
     2. LỌC + BẢNG DỮ LIỆU
     ============================================================ */
  async function loadMeta() {
    if (state.meta) return state.meta;
    state.meta = await Api.meta();
    return state.meta;
  }

  function buildFilterUI() {
    const m = state.meta; if (!m) return;
    const html = [];

    // numeric ranges
    Object.entries(m.numeric_ranges).forEach(([col, r]) => {
      html.push(`<div class="filter-field">
        <label>${col} (${r.min} – ${r.max})</label>
        <div class="range-row">
          <input type="number" id="f_${col}_min" placeholder="min" />
          <input type="number" id="f_${col}_max" placeholder="max" />
        </div>
      </div>`);
    });

    // categories
    Object.entries(m.categories).forEach(([col, vals]) => {
      html.push(`<div class="filter-field">
        <label>${col}</label>
        <div class="chip-group" data-cat="${col}">
          ${vals.map(v => `<button type="button" class="chip" data-val="${v}">${v}</button>`).join("")}
        </div>
      </div>`);
    });

    // search
    html.push(`<div class="filter-field">
      <label>Tìm kiếm (toàn văn)</label>
      <input type="text" id="f_search" placeholder="Nhập từ khoá…" />
    </div>`);

    $("filterGrid").innerHTML = html.join("");

    // chip toggle
    document.querySelectorAll(".chip-group").forEach(g => {
      g.addEventListener("click", (ev) => {
        const chip = ev.target.closest(".chip"); if (!chip) return;
        chip.classList.toggle("is-on");
      });
    });
  }

  function collectFilters() {
    const f = {};
    if (!state.meta) return f;
    Object.keys(state.meta.numeric_ranges).forEach(col => {
      const lo = $("f_"+col+"_min"), hi = $("f_"+col+"_max");
      if (lo && lo.value !== "") f[col+"_min"] = Number(lo.value);
      if (hi && hi.value !== "") f[col+"_max"] = Number(hi.value);
    });
    document.querySelectorAll(".chip-group").forEach(g => {
      const col = g.dataset.cat;
      const picked = Array.from(g.querySelectorAll(".chip.is-on")).map(c => c.dataset.val);
      if (picked.length) f[col] = picked;
    });
    const s = $("f_search"); if (s && s.value.trim()) f.search = s.value.trim();
    return f;
  }

  async function loadDataTable() {
    try {
      await loadMeta();
      const payload = {filters: state.filters, page: state.page, page_size: state.pageSize,
        sort_by: state.sortBy, sort_dir: state.sortDir};
      const d = await Api.data(payload);
      $("dataCount").textContent = `${d.total} dòng · trang ${d.page}/${d.pages}`;
      buildTable("dataTable", d.columns, d.rows);

      // header click sort
      const tbl = $("dataTable").querySelector("table");
      if (tbl) tbl.querySelectorAll("th").forEach((th, idx) => {
        th.addEventListener("click", () => {
          const col = d.columns[idx];
          state.sortBy = col;
          state.sortDir = state.sortBy === col && state.sortDir === "asc" ? "desc" : "asc";
          state.page = 1; loadDataTable();
        });
      });

      $("pgInfo").textContent = `Trang ${d.page} / ${d.pages} · tổng ${d.total}`;
      $("pgPrev").disabled = d.page <= 1;
      $("pgNext").disabled = d.page >= d.pages;
      state._pages = d.pages;
    } catch(e) { $("dataTable").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ============================================================
     3. BỆNH NHÂN
     ============================================================ */
  async function loadPatients() {
    try {
      const d = await Api.patients({filters: state.filters});
      $("ptStats").innerHTML =
        statCard("Tổng bệnh nhân", d.total, "primary") +
        statCard("Tuổi TB", num(d.age_stats.mean, 1), "violet") +
        statCard("Trung vị tuổi", num(d.age_stats.median, 1)) +
        statCard("Min tuổi", num(d.age_stats.min, 0)) +
        statCard("Max tuổi", num(d.age_stats.max, 0));

      Charts.make("ptAge", {
        type:"bar",
        data:{labels:Object.keys(d.age_hist),
          datasets:[{label:"Số bệnh nhân", data:Object.values(d.age_hist), backgroundColor:Charts.palette(0)}]},
        options:Charts.baseOpts()
      });
      Charts.make("ptRace", {
        type:"pie",
        data:{labels:Object.keys(d.race_dist),
          datasets:[{data:Object.values(d.race_dist),
            backgroundColor:["#2563eb","#7c3aed","#dc2626","#16a34a","#d97706"]}]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{position:"bottom", labels:{color:Charts.textColor()}}}}
      });
      Charts.make("ptMarital", {
        type:"bar",
        data:{labels:Object.keys(d.marital_dist),
          datasets:[{label:"Số bệnh nhân", data:Object.values(d.marital_dist), backgroundColor:Charts.palette(1)}]},
        options:{...Charts.baseOpts(), indexAxis:"y"}
      });

      const rbs = d.race_by_status || {};
      const races = Object.keys(rbs);
      const statuses = [...new Set(races.flatMap(r => Object.keys(rbs[r] || {})))];
      Charts.make("ptRaceStatus", {
        type:"bar",
        data:{labels:races, datasets:statuses.map((s, i) => ({
          label:s, data:races.map(r => (rbs[r] && rbs[r][s]) || 0),
          backgroundColor:Charts.palette(i)
        }))},
        options:{...Charts.baseOpts(), scales:{x:{stacked:true}, y:{stacked:true, beginAtZero:true}}}
      });
    } catch(e) { $("ptStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ============================================================
     4. BỆNH & KHỐI U
     ============================================================ */
  async function loadDisease() {
    try {
      const d = await Api.disease({filters: state.filters});
      $("disStats").innerHTML =
        statCard("Tổng", d.total, "primary") +
        statCard("Size TB (mm)", num(d.size_stats.mean, 1), "violet") +
        statCard("Trung vị", num(d.size_stats.median, 1)) +
        statCard("Min", num(d.size_stats.min, 0)) +
        statCard("Max", num(d.size_stats.max, 0));

      const common = Charts.baseOpts();
      Charts.make("disSize", {type:"bar", data:{labels:Object.keys(d.size_hist),
        datasets:[{label:"Số ca", data:Object.values(d.size_hist), backgroundColor:Charts.palette(0)}]},
        options:common});
      Charts.make("disStage", {type:"bar", data:{labels:Object.keys(d.stage_dist),
        datasets:[{label:"Số ca", data:Object.values(d.stage_dist), backgroundColor:Charts.palette(1)}]},
        options:common});
      Charts.make("disGrade", {type:"bar", data:{labels:Object.keys(d.grade_dist),
        datasets:[{label:"Số ca", data:Object.values(d.grade_dist), backgroundColor:Charts.palette(3)}]},
        options:common});
      Charts.make("disDiff", {type:"bar", data:{labels:Object.keys(d.differentiate_dist),
        datasets:[{label:"Số ca", data:Object.values(d.differentiate_dist), backgroundColor:Charts.palette(4)}]},
        options:{...Charts.baseOpts(), indexAxis:"y"}});
      Charts.make("disSizeStage", {type:"bar", data:{labels:Object.keys(d.avg_size_by_stage),
        datasets:[{label:"Size TB (mm)", data:Object.values(d.avg_size_by_stage), backgroundColor:Charts.palette(5)}]},
        options:common});
      Charts.make("disSizeGrade", {type:"bar", data:{labels:Object.keys(d.avg_size_by_grade),
        datasets:[{label:"Size TB (mm)", data:Object.values(d.avg_size_by_grade), backgroundColor:Charts.palette(2)}]},
        options:common});
    } catch(e) { $("disStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ============================================================
     5. SINH HỌC & HẠCH
     ============================================================ */
  async function loadBiology() {
    try {
      const d = await Api.biology({filters: state.filters});
      $("bioStats").innerHTML =
        statCard("Tổng", d.total, "primary") +
        statCard("Hạch dương tính TB", num(d.node_stats.mean_positive, 2), "danger") +
        statCard("Hạch kiểm tra TB", num(d.node_stats.mean_examined, 1)) +
        statCard("Max hạch (+)", num(d.node_stats.max_positive, 0));

      Charts.make("bioEst", {type:"doughnut", data:{labels:Object.keys(d.estrogen_dist),
        datasets:[{data:Object.values(d.estrogen_dist),
          backgroundColor:["#2563eb","#dc2626"]}]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{position:"bottom", labels:{color:Charts.textColor()}}}}});
      Charts.make("bioProg", {type:"doughnut", data:{labels:Object.keys(d.progesterone_dist),
        datasets:[{data:Object.values(d.progesterone_dist),
          backgroundColor:["#7c3aed","#dc2626"]}]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{position:"bottom", labels:{color:Charts.textColor()}}}}});

      const ep = d.ep_crosstab || {};
      const estLabels = Object.keys(ep);
      const progLabels = [...new Set(estLabels.flatMap(e => Object.keys(ep[e] || {})))];
      Charts.make("bioEP", {type:"bar", data:{labels:estLabels,
        datasets:progLabels.map((p, i) => ({
          label:p, data:estLabels.map(e => (ep[e] && ep[e][p]) || 0),
          backgroundColor:Charts.palette(i)}))},
        options:{...Charts.baseOpts(), scales:{x:{stacked:true}, y:{stacked:true, beginAtZero:true}}}});

      Charts.make("bioNode", {type:"bar", data:{labels:Object.keys(d.node_hist),
        datasets:[{label:"Số ca", data:Object.values(d.node_hist), backgroundColor:Charts.palette(3)}]},
        options:Charts.baseOpts()});

      Charts.make("bioSurvEst", {type:"bar", data:{labels:Object.keys(d.survival_by_estrogen),
        datasets:[{label:"Tháng TB", data:Object.values(d.survival_by_estrogen), backgroundColor:Charts.palette(0)}]},
        options:Charts.baseOpts()});
      Charts.make("bioSurvProg", {type:"bar", data:{labels:Object.keys(d.survival_by_progesterone),
        datasets:[{label:"Tháng TB", data:Object.values(d.survival_by_progesterone), backgroundColor:Charts.palette(1)}]},
        options:Charts.baseOpts()});
    } catch(e) { $("bioStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ============================================================
     6. SỐNG CÒN & THỐNG KÊ
     ============================================================ */
  async function loadSurvival() {
    try {
      const d = await Api.survival({filters: state.filters});
      $("svStats").innerHTML =
        statCard("Tổng", d.total, "primary") +
        statCard("Tháng TB", num(d.survival_stats.mean, 1), "violet") +
        statCard("Trung vị", num(d.survival_stats.median, 1)) +
        statCard("Min", num(d.survival_stats.min, 0)) +
        statCard("Max", num(d.survival_stats.max, 0));

      Charts.make("svHist", {type:"bar", data:{labels:Object.keys(d.survival_hist),
        datasets:[{label:"Số ca", data:Object.values(d.survival_hist), backgroundColor:Charts.palette(0)}]},
        options:Charts.baseOpts()});
      Charts.make("svStage", {type:"bar", data:{labels:Object.keys(d.survival_by_stage),
        datasets:[{label:"Tháng TB", data:Object.values(d.survival_by_stage), backgroundColor:Charts.palette(1)}]},
        options:Charts.baseOpts()});
      Charts.make("svGrade", {type:"bar", data:{labels:Object.keys(d.survival_by_grade),
        datasets:[{label:"Tháng TB", data:Object.values(d.survival_by_grade), backgroundColor:Charts.palette(2)}]},
        options:Charts.baseOpts()});
      Charts.make("svStatus", {type:"bar", data:{labels:Object.keys(d.survival_by_status),
        datasets:[{label:"Tháng TB", data:Object.values(d.survival_by_status), backgroundColor:Charts.palette(3)}]},
        options:Charts.baseOpts()});

      // correlation heatmap
      const cols = d.correlation_columns || [];
      if (cols.length) {
        const datasets = [];
        cols.forEach((row, i) => {
          cols.forEach((col, j) => {
            const v = d.correlation[row]?.[col] ?? 0;
            datasets.push({x:col, y:row, v:v});
          });
        });
        const ctx = $("svCorr").getContext("2d");
        Charts.destroy("svCorr");
        const c = new Chart(ctx, {
          type:"matrix",
          data:{datasets:[{data:datasets, backgroundColor:(ctx2) => {
            const v = ctx2.raw?.v ?? 0;
            const a = Math.abs(v);
            return v >= 0 ? `rgba(37,99,235,${a})` : `rgba(220,38,38,${a})`;
          }, width:({chart}) => (chart.chartArea||{width:400}).width / cols.length - 2,
             height:({chart}) => (chart.chartArea||{height:400}).height / cols.length - 2}]},
          options:{responsive:true,maintainAspectRatio:false,
            plugins:{legend:{display:false}, tooltip:{callbacks:{title:()=>"", label:(c)=>{
              const r = c.raw; return `${r.y} × ${r.x}: ${r.v}`;}}}},
            scales:{x:{type:"category", labels:cols, ticks:{color:Charts.textColor(), font:{size:9}}},
                    y:{type:"category", labels:cols, ticks:{color:Charts.textColor(), font:{size:9}}}}}
        });
      }

      // descriptive table
      const descCols = ["column","count","mean","std","min","q25","median","q75","max"];
      const descRows = Object.entries(d.descriptive).map(([c, s]) => ({
        column:c, count:s.count, mean:num(s.mean,2), std:num(s.std,2), min:num(s.min,2),
        q25:num(s.q25,2), median:num(s.median,2), q75:num(s.q75,2), max:num(s.max,2)
      }));
      buildTable("svDesc", descCols, descRows);
    } catch(e) { $("svStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ============================================================
     7. ML
     ============================================================ */
  function buildMlForm() {
    const html = `
      <div class="filter-field"><label>Tuổi</label><input type="number" id="ml_age" value="55"></div>
      <div class="filter-field"><label>Chủng tộc</label><select id="ml_race">
        <option>White</option><option>Black</option><option>Other</option></select></div>
      <div class="filter-field"><label>Hôn nhân</label><select id="ml_marital_status">
        <option>Married</option><option>Single</option><option>Divorced</option>
        <option>Widowed</option><option>Separated</option></select></div>
      <div class="filter-field"><label>T Stage</label><select id="ml_t_stage">
        <option>T1</option><option selected>T2</option><option>T3</option><option>T4</option></select></div>
      <div class="filter-field"><label>N Stage</label><select id="ml_n_stage">
        <option>N1</option><option>N2</option><option>N3</option></select></div>
      <div class="filter-field"><label>6th Stage</label><select id="ml_stage_6th">
        <option>IIA</option><option selected>IIB</option><option>IIIA</option>
        <option>IIIB</option><option>IIIC</option></select></div>
      <div class="filter-field"><label>Differentiate</label><select id="ml_differentiate">
        <option>Well differentiated</option><option selected>Moderately differentiated</option>
        <option>Poorly differentiated</option><option>Undifferentiated</option></select></div>
      <div class="filter-field"><label>Grade</label><select id="ml_grade">
        <option>1</option><option selected>2</option><option>3</option><option>4</option></select></div>
      <div class="filter-field"><label>A Stage</label><select id="ml_a_stage">
        <option selected>Regional</option><option>Distant</option></select></div>
      <div class="filter-field"><label>Tumor Size (mm)</label><input type="number" id="ml_tumor_size" value="30"></div>
      <div class="filter-field"><label>Estrogen</label><select id="ml_estrogen_status">
        <option selected>Positive</option><option>Negative</option></select></div>
      <div class="filter-field"><label>Progesterone</label><select id="ml_progesterone_status">
        <option selected>Positive</option><option>Negative</option></select></div>
      <div class="filter-field"><label>Hạch kiểm tra</label><input type="number" id="ml_regional_node_examined" value="10"></div>
      <div class="filter-field"><label>Hạch dương tính</label><input type="number" id="ml_regional_node_positive" value="2"></div>
      <div class="filter-field"><label>Tháng sống</label><input type="number" id="ml_survival_months" value="60"></div>
      <div class="filter-field"><label>Mô hình</label><select id="ml_model">
        <option>Random Forest</option><option>Logistic Regression</option>
        <option>Gradient Boosting</option><option>Decision Tree</option></select></div>
    `;
    $("mlForm").innerHTML = html;
  }

  function readMlForm() {
    const v = id => document.getElementById(id)?.value;
    return {
      age:Number(v("ml_age")), race:v("ml_race"), marital_status:v("ml_marital_status"),
      t_stage:v("ml_t_stage"), n_stage:v("ml_n_stage"), stage_6th:v("ml_stage_6th"),
      differentiate:v("ml_differentiate"), grade:Number(v("ml_grade")), a_stage:v("ml_a_stage"),
      tumor_size:Number(v("ml_tumor_size")), estrogen_status:v("ml_estrogen_status"),
      progesterone_status:v("ml_progesterone_status"),
      regional_node_examined:Number(v("ml_regional_node_examined")),
      regional_node_positive:Number(v("ml_regional_node_positive")),
      survival_months:Number(v("ml_survival_months")), model:v("ml_model")
    };
  }

  async function trainModels() {
    try {
      $("mlStats").innerHTML = `<div class="alert" style="background:var(--primary-soft);color:var(--primary);border-color:var(--primary)">⏳ Đang huấn luyện…</div>`;
      const d = await Api.train();
      state.ml = d;
      renderMl(d);
    } catch(e) { $("mlStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  function renderMl(d) {
    const best = Object.entries(d.metrics).sort((a,b) => b[1].auc - a[1].auc)[0];
    $("mlStats").innerHTML =
      statCard("Số đặc trưng", d.feature_names.length, "primary") +
      statCard("Alive (train)", d.class_balance.alive, "success") +
      statCard("Dead (train)", d.class_balance.dead, "danger") +
      statCard("Best model", best[0], "violet") +
      statCard("AUC tốt nhất", best[1].auc);

    const names = Object.keys(d.metrics);
    const metrics = ["accuracy","precision","recall","f1","auc"];
    Charts.make("mlBar", {
      type:"bar",
      data:{labels:names, datasets:metrics.map((m, i) => ({
        label:m, data:names.map(n => d.metrics[n][m]), backgroundColor:Charts.palette(i)}))},
      options:{...Charts.baseOpts()}
    });

    Charts.make("mlRoc", {
      type:"line",
      data:{datasets:names.map((n, i) => ({
        label:`${n} (AUC=${d.metrics[n].auc})`,
        data:d.metrics[n].roc.fpr.map((x, k) => ({x:x, y:d.metrics[n].roc.tpr[k]})),
        borderColor:Charts.palette(i), backgroundColor:Charts.hexA(Charts.palette(i), .15),
        tension:.2, pointRadius:0, fill:false
      })).concat([{label:"Đường chéo", data:[{x:0,y:0},{x:1,y:1}],
        borderColor:"#94a3b8", borderDash:[4,4], pointRadius:0, fill:false}])},
      options:{responsive:true, maintainAspectRatio:false,
        parsing:false,
        plugins:{legend:{position:"bottom", labels:{color:Charts.textColor(), font:{size:10}}}},
        scales:{x:{type:"linear", min:0, max:1, ticks:{color:Charts.textColor()}},
                y:{type:"linear", min:0, max:1, ticks:{color:Charts.textColor()}}}}
    });

    const fi = (d.feature_importance || []).slice(0, 12);
    Charts.make("mlFeat", {
      type:"bar",
      data:{labels:fi.map(x => x.feature), datasets:[{label:"Importance",
        data:fi.map(x => x.importance), backgroundColor:Charts.palette(1)}]},
      options:{...Charts.baseOpts(), indexAxis:"y"}
    });

    const cm = best[1].confusion_matrix;
    Charts.make("mlCm", {
      type:"matrix",
      data:{datasets:[{data:[
        {x:"Dự Alive", y:"Thực Alive", v:cm[0][0]},
        {x:"Dự Dead", y:"Thực Alive", v:cm[0][1]},
        {x:"Dự Alive", y:"Thực Dead", v:cm[1][0]},
        {x:"Dự Dead", y:"Thực Dead", v:cm[1][1]}
      ], backgroundColor:(c) => {
        const v = c.raw?.v ?? 0; const max = Math.max(cm[0][0], cm[0][1], cm[1][0], cm[1][1]) || 1;
        return `rgba(37,99,235,${0.15 + 0.75*(v/max)})`;
      }, width:90, height:60}]},
      options:{responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}, tooltip:{callbacks:{
          label:(c) => `${c.raw.y} / ${c.raw.x}: ${c.raw.v}`}}},
        scales:{x:{type:"category", labels:["Dự Alive","Dự Dead"], ticks:{color:Charts.textColor()}},
                y:{type:"category", labels:["Thực Alive","Thực Dead"], ticks:{color:Charts.textColor()}}}}
    });
  }

  async function compareModels() {
    try {
      const d = await Api.compare();
      const rows = d.comparison.map(r => ({
        model:r.model, accuracy:r.accuracy, precision:r.precision,
        recall:r.recall, f1:r.f1, auc:r.auc
      }));
      state.ml = state.ml || {};
      state.ml.comparison = rows;
      // hiển thị bảng bên dưới ROC
      const container = document.getElementById("mlResult");
      container.innerHTML = `<div class="table-wrap"><table class="dt">
        <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead>
        <tbody>${rows.map(r => `<tr><td><strong>${r.model}</strong></td>
          <td class="mono">${r.accuracy}</td><td class="mono">${r.precision}</td>
          <td class="mono">${r.recall}</td><td class="mono">${r.f1}</td>
          <td class="mono"><strong>${r.auc}</strong></td></tr>`).join("")}</tbody>
        </table></div>
        <p class="muted" style="margin-top:8px">🏆 Best model (theo AUC): <strong>${d.best}</strong></p>`;
    } catch(e) { document.getElementById("mlResult").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  async function predictPatient() {
    try {
      const payload = readMlForm();
      const d = await Api.mlPredict(payload);
      const cls = d.prediction === "Dead" ? "danger" : "success";
      document.getElementById("mlResult").innerHTML = `
        <div class="result__card" style="padding:0">
          <div class="stat-grid">
            ${statCard("Mô hình", d.model, "primary")}
            ${statCard("Dự đoán", d.prediction, cls)}
            ${statCard("P(Dead)", d.probability_dead, "danger")}
            ${statCard("P(Alive)", d.probability_alive, "success")}
          </div>
        </div>`;
    } catch(e) { document.getElementById("mlResult").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ============================================================
     LOAD DISPATCH
     ============================================================ */
  async function load(name) {
    try {
      if (name === "overview") { await loadOverview(); }
      else if (name === "data") { await loadMeta(); buildFilterUI(); await loadDataTable(); }
      else if (name === "patients") { await loadPatients(); }
      else if (name === "disease") { await loadDisease(); }
      else if (name === "biology") { await loadBiology(); }
      else if (name === "survival") { await loadSurvival(); }
      else if (name === "ml") { buildMlForm(); if (!state.ml) await trainModels(); }
    } catch(e) { console.error("[load]", name, e); }
  }

  /* ============================================================
     EVENT BINDINGS
     ============================================================ */
  function bind() {
    $("btnRefreshOverview").addEventListener("click", loadOverview);

    $("btnApplyFilter").addEventListener("click", () => {
      state.filters = collectFilters(); state.page = 1; loadDataTable();
    });
    $("btnClearFilter").addEventListener("click", () => {
      state.filters = {}; state.page = 1;
      document.querySelectorAll("#filterGrid input").forEach(i => i.value = "");
      document.querySelectorAll("#filterGrid .chip.is-on").forEach(c => c.classList.remove("is-on"));
      loadDataTable();
    });
    $("pgPrev").addEventListener("click", () => { if (state.page > 1) { state.page--; loadDataTable(); } });
    $("pgNext").addEventListener("click", () => {
      if (state.page < (state._pages || 1)) { state.page++; loadDataTable(); }
    });

    document.querySelectorAll("[data-refresh]").forEach(b => {
      b.addEventListener("click", () => load(b.dataset.refresh));
    });

    $("btnTrain").addEventListener("click", trainModels);
    $("btnCompare").addEventListener("click", async () => { await compareModels(); });
    $("btnPredictPatient").addEventListener("click", predictPatient);
  }

  return { load, bind, state, trainModels };
})();

/* ============================================================
   MODULE 7 · APP (SVM gốc)
   ============================================================ */
const App = (() => {
  async function handlePredict(event) {
    event.preventDefault();
    let payload;
    try { payload = UI.readForm(); }
    catch (err) { UI.showError(err.message); return; }
    UI.setLoading(true);
    try {
      const data = await Api.predict(payload);
      UI.showResult(data); UI.setStatus("online");
    } catch (err) {
      const hint = err.name === "AbortError" ? "Yêu cầu quá thời gian (timeout)." : err.message;
      UI.showError(`${hint}<br><br>Kiểm tra lại: FastAPI đang chạy? Model đã load chưa?`);
      UI.setStatus("offline", hint);
    } finally { UI.setLoading(false); }
  }

  async function checkHealth() {
    UI.setStatus("checking");
    try {
      const h = await Api.health();
      if (h && h.model_loaded) UI.setStatus("online");
      else UI.setStatus("offline", h && h.error ? h.error : "Model chưa sẵn sàng.");
    } catch { UI.setStatus("offline"); }
  }

  function bindSvmEvents() {
    document.getElementById("form").addEventListener("submit", handlePredict);
    document.getElementById("btnSampleMalignant").addEventListener("click", () => {
      UI.fillSample("malignant"); UI.clearResult();
    });
    document.getElementById("btnSampleBenign").addEventListener("click", () => {
      UI.fillSample("benign"); UI.clearResult();
    });
    document.getElementById("btnReset").addEventListener("click", () => {
      setTimeout(() => {
        document.querySelectorAll("input.is-invalid").forEach(el => el.classList.remove("is-invalid"));
        UI.clearResult();
      }, 0);
    });
    document.getElementById("btnTheme").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      UI.applyTheme(next);
    });
  }

  function init() {
    let saved = "light";
    try { saved = localStorage.getItem("bc-theme") || "light"; } catch {}
    UI.applyTheme(saved);

    UI.buildForm();
    bindSvmEvents();
    Tabs.init();
    Dashboard.bind();
    checkHealth();
  }
  return { init };
})();

document.addEventListener("DOMContentLoaded", App.init);
</script>
</body>
</html>
"""


# ==============================================================
# 5. ENTRYPOINT
# ==============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
