from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import pandas as pd
import io
import math
import numpy as np
import joblib
import uvicorn

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

if _df is not None and "status" in _df.columns:
    _df["status"] = _df["status"].astype(str).str.strip().str.title()


# ==============================================================
# 1c. HELPERS
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
    for col in NUM_COLS:
        lo = f.get(f"{col}_min")
        hi = f.get(f"{col}_max")
        if lo is not None:
            d = d[d[col] >= lo]
        if hi is not None:
            d = d[d[col] <= hi]
    for col in CAT_COLS + ["status"]:
        vals = f.get(col)
        if vals:
            d = d[d[col].isin(vals)]
    q = (f.get("search") or "").strip().lower()
    if q:
        mask = np.zeros(len(d), dtype=bool)
        for c in d.columns:
            mask |= d[c].astype(str).str.lower().str.contains(q, na=False).to_numpy()
        d = d[mask]
    return d


def _safe(v):
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


_ml_state: Dict[str, Any] = {
    "trained": False, "metrics": {}, "models": {}, "scaler": None,
    "encoders": {}, "feature_names": [], "X_test": None, "y_test": None,
    "y_pred": {}, "y_prob": {}, "error": None,
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
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
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

        rf = candidates["Random Forest"]
        importances = sorted(
            [{"feature": f, "importance": round(float(v), 4)}
             for f, v in zip(feat_names, rf.feature_importances_)],
            key=lambda x: -x["importance"],
        )

        _ml_state.update({
            "trained": True, "metrics": metrics, "models": candidates,
            "scaler": scaler, "encoders": encoders, "feature_names": feat_names,
            "X_test": X_te, "y_test": y_te.tolist(),
            "y_pred": y_pred_map, "y_prob": y_prob_map,
            "feature_importance": importances,
            "class_balance": {"alive": int((y == 0).sum()), "dead": int((y == 1).sum())},
            "error": None,
        })
    except Exception as exc:
        _ml_state["error"] = str(exc)
        _ml_state["trained"] = False
    return _ml_state


# ==============================================================
# 2. SCHEMAS
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
    title="Breast Cancer Analytics API",
    description="Dashboard phân tích sống còn + SVM chẩn đoán 30 đặc trưng.",
    version="2.0.0",
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


@app.post("/predict")
def predict(data: BreastCancerInput):
    if _model is None:
        raise HTTPException(status_code=503,
                            detail=f"Mô hình chưa sẵn sàng. {_model_error or ''}".strip())
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
    if _df is None:
        return {"columns": [], "categories": {}, "numeric_ranges": {}}
    categories = {}
    for c in CAT_COLS + ["status"]:
        if c in _df.columns:
            categories[c] = sorted(_df[c].dropna().astype(str).unique().tolist())
    numeric_ranges = {}
    for c in NUM_COLS:
        if c in _df.columns:
            numeric_ranges[c] = {"min": _safe(_df[c].min()), "max": _safe(_df[c].max())}
    return {
        "columns": _df.columns.tolist(),
        "categories": categories,
        "numeric_ranges": numeric_ranges,
        "rows": int(len(_df)),
        "error": _df_error,
    }


# ---------- TỔNG QUAN ----------
@app.get("/api/overview")
def api_overview():
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    df = _df
    total = int(df.shape[0])
    alive = int((df["status"] == "Alive").sum()) if "status" in df.columns else 0
    dead = int((df["status"] == "Dead").sum()) if "status" in df.columns else 0
    avg_surv = float(df["survival_months"].mean()) if "survival_months" in df.columns else 0.0
    median_surv = float(df["survival_months"].median()) if "survival_months" in df.columns else 0.0
    stage_dist = (df["stage_6th"].value_counts().to_dict()
                  if "stage_6th" in df.columns else {})
    status_dist = (df["status"].value_counts().to_dict()
                   if "status" in df.columns else {})

    return {
        "rows": total,
        "cols": int(df.shape[1]),
        "alive": alive,
        "dead": dead,
        "alive_pct": round(100.0 * alive / total, 1) if total else 0,
        "dead_pct": round(100.0 * dead / total, 1) if total else 0,
        "avg_survival": round(avg_surv, 1),
        "median_survival": round(median_surv, 1),
        "stage_distribution": {str(k): int(v) for k, v in stage_dist.items()},
        "target_distribution": {str(k): int(v) for k, v in status_dist.items()},
        "columns": df.columns.tolist(),
    }


# ---------- KHÁM PHÁ DỮ LIỆU ----------
@app.post("/api/data")
def api_data(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    if payload.sort_by and payload.sort_by in d.columns:
        d = d.sort_values(payload.sort_by,
                          ascending=(payload.sort_dir.lower() != "desc"),
                          kind="mergesort")

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
        "total": total, "page": page, "page_size": size,
        "pages": max(1, math.ceil(total / size)),
        "rows": rows, "columns": d.columns.tolist(),
    }


# ---------- ĐẶC ĐIỂM BỆNH NHÂN ----------
@app.post("/api/analysis/patients")
def api_patients(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    age_bins = [0, 30, 40, 50, 60, 70, 80, 200]
    age_labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    age_hist = (pd.cut(d["age"], bins=age_bins, labels=age_labels, right=False)
                .value_counts().reindex(age_labels).fillna(0).astype(int).to_dict())

    def _dist(col):
        return {str(k): int(v) for k, v in d[col].value_counts().to_dict().items()}

    marital_by_status = {}
    if "status" in d.columns:
        ct = pd.crosstab(d["marital_status"], d["status"])
        marital_by_status = {str(k): {str(kk): int(vv) for kk, vv in v.items()}
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
        "marital_by_status": marital_by_status,
    }


# ---------- BỆNH & KHỐI U ----------
@app.post("/api/analysis/disease")
def api_disease(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    size_bins = [0, 10, 20, 30, 40, 50, 75, 100, 1000]
    size_labels = ["<10", "10-19", "20-29", "30-39", "40-49", "50-74", "75-99", "100+"]
    size_hist = (pd.cut(d["tumor_size"], bins=size_bins, labels=size_labels, right=False)
                 .value_counts().reindex(size_labels).fillna(0).astype(int).to_dict())

    def _dist(col):
        return {str(k): int(v) for k, v in d[col].value_counts().to_dict().items()}

    avg_size_by_stage = (d.groupby("stage_6th")["tumor_size"].mean().round(2).to_dict()
                         if "stage_6th" in d.columns else {})
    avg_size_by_grade = (d.groupby("grade")["tumor_size"].mean().round(2).to_dict()
                         if "grade" in d.columns else {})

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
        "avg_size_by_stage": {str(k): float(v) for k, v in avg_size_by_stage.items()},
        "avg_size_by_grade": {str(k): float(v) for k, v in avg_size_by_grade.items()},
    }


# ---------- YẾU TỐ SINH HỌC ----------
@app.post("/api/analysis/biology")
def api_biology(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    def _dist(col):
        return {str(k): int(v) for k, v in d[col].value_counts().to_dict().items()}

    # Có phân biệt Alive/Dead cho Estrogen/Progesterone
    est_by_status, prog_by_status = {}, {}
    if "status" in d.columns:
        ct = pd.crosstab(d["estrogen_status"], d["status"])
        est_by_status = {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                         for k, v in ct.to_dict(orient="index").items()}
        ct = pd.crosstab(d["progesterone_status"], d["status"])
        prog_by_status = {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                          for k, v in ct.to_dict(orient="index").items()}

    node_bins = [-1, 0, 1, 3, 5, 10, 100]
    node_labels = ["0", "1", "2-3", "4-5", "6-10", "10+"]
    node_hist = (pd.cut(d["regional_node_positive"], bins=node_bins, labels=node_labels)
                 .value_counts().reindex(node_labels).fillna(0).astype(int).to_dict())

    avg_nodes_by_stage = (d.groupby("stage_6th")["regional_node_positive"].mean().round(2).to_dict()
                          if "stage_6th" in d.columns else {})
    surv_by_est = (d.groupby("estrogen_status")["survival_months"].mean().round(2).to_dict()
                   if "estrogen_status" in d.columns else {})
    surv_by_prog = (d.groupby("progesterone_status")["survival_months"].mean().round(2).to_dict()
                    if "progesterone_status" in d.columns else {})

    return {
        "total": int(len(d)),
        "estrogen_dist": _dist("estrogen_status"),
        "progesterone_dist": _dist("progesterone_status"),
        "estrogen_by_status": est_by_status,
        "progesterone_by_status": prog_by_status,
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


# ---------- SỐNG CÒN ----------
@app.post("/api/analysis/survival")
def api_survival(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    bins = list(range(0, 121, 10))
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
    surv_hist = (pd.cut(d["survival_months"], bins=bins, labels=labels, right=False)
                 .value_counts().reindex(labels).fillna(0).astype(int).to_dict())

    # Boxplot theo stage
    boxplot_by_stage = {}
    if "stage_6th" in d.columns:
        for stage, grp in d.groupby("stage_6th"):
            s = grp["survival_months"].dropna()
            if len(s) > 0:
                boxplot_by_stage[str(stage)] = {
                    "min": float(s.min()),
                    "q1": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "q3": float(s.quantile(0.75)),
                    "max": float(s.max()),
                }

    # Scatter (giới hạn 400 điểm)
    sample = d.sample(n=min(400, len(d)), random_state=42) if len(d) > 0 else d
    scatter_tumor = []
    scatter_node = []
    for _, row in sample.iterrows():
        st = str(row.get("status", ""))
        scatter_tumor.append({
            "x": _safe(row["tumor_size"]),
            "y": _safe(row["survival_months"]),
            "status": st,
        })
        scatter_node.append({
            "x": _safe(row["regional_node_positive"]),
            "y": _safe(row["survival_months"]),
            "status": st,
        })

    numeric_df = d.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(3).fillna(0).to_dict() if len(numeric_df.columns) else {}

    desc = {}
    for c in numeric_df.columns:
        s = numeric_df[c]
        desc[c] = {
            "count": int(s.count()), "mean": _safe(s.mean()), "std": _safe(s.std()),
            "min": _safe(s.min()), "q25": _safe(s.quantile(0.25)),
            "median": _safe(s.median()), "q75": _safe(s.quantile(0.75)), "max": _safe(s.max()),
        }

    by_stage = (d.groupby("stage_6th")["survival_months"].mean().round(2).to_dict()
                if "stage_6th" in d.columns else {})
    by_grade = (d.groupby("grade")["survival_months"].mean().round(2).to_dict()
                if "grade" in d.columns else {})
    by_status = (d.groupby("status")["survival_months"].mean().round(2).to_dict()
                 if "status" in d.columns else {})

    return {
        "total": int(len(d)),
        "survival_hist": surv_hist,
        "survival_stats": {
            "mean": _safe(d["survival_months"].mean()),
            "median": _safe(d["survival_months"].median()),
            "min": _safe(d["survival_months"].min()),
            "max": _safe(d["survival_months"].max()),
        },
        "survival_boxplot_by_stage": boxplot_by_stage,
        "scatter_tumor_survival": scatter_tumor,
        "scatter_node_survival": scatter_node,
        "correlation": corr,
        "correlation_columns": numeric_df.columns.tolist(),
        "descriptive": desc,
        "survival_by_stage": {str(k): float(v) for k, v in by_stage.items()},
        "survival_by_grade": {str(k): float(v) for k, v in by_grade.items()},
        "survival_by_status": {str(k): float(v) for k, v in by_status.items()},
    }


# ---------- ML ----------
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
                X_row[c] = 0
        else:
            X_row[c] = pd.to_numeric(X_row[c], errors="coerce").fillna(0.0)
    X_row = X_row[feat_names]
    mdl = _ml_state["models"][model_name]
    X_in = _ml_state["scaler"].transform(X_row) if model_name == "Logistic Regression" else X_row
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
            "model": name, "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "auc": m["auc"],
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
<title>Phân tích dữ liệu bệnh nhân ung thư vú</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@2.0.1/dist/chartjs-chart-matrix.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@sgratzl/chartjs-chart-boxplot@4.4.1/build/index.umd.min.js"></script>
<style>
/* ============================================================
   Màu sắc: chỉ 4 tông chính
   🔵 xanh dương → thông tin chung
   🟢 xanh lá    → Alive / Positive
   🔴 đỏ         → Dead / Negative
   ⚪ xám        → nền / phụ trợ
   ============================================================ */
:root{
  --blue:#2563eb;   --blue-dark:#1d4ed8;  --blue-soft:#dbeafe;
  --green:#16a34a;  --green-dark:#15803d; --green-soft:#dcfce7;
  --red:#dc2626;    --red-dark:#b91c1c;   --red-soft:#fee2e2;
  --gray:#64748b;   --gray-dark:#475569;  --gray-soft:#f1f5f9;

  --bg:#f8fafc;--surface:#fff;--surface-2:#f8fafc;--surface-3:#f1f5f9;
  --text:#0f172a;--text-muted:#64748b;--text-soft:#94a3b8;
  --border:#e2e8f0;--border-strong:#cbd5e1;

  --radius:12px;--radius-lg:16px;
  --shadow-sm:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04);
  --shadow:0 4px 12px rgba(15,23,42,.08),0 2px 4px rgba(15,23,42,.04);
  --font:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --transition:180ms cubic-bezier(0.4,0,0.2,1);
}
[data-theme="dark"]{
  --bg:#020617;--surface:#0f172a;--surface-2:#111c33;--surface-3:#1e293b;
  --text:#f1f5f9;--text-muted:#94a3b8;--text-soft:#64748b;
  --border:#1e293b;--border-strong:#334155;
  --blue:#60a5fa;--blue-soft:rgba(96,165,250,.14);
  --green:#4ade80;--green-soft:rgba(74,222,128,.14);
  --red:#f87171;--red-soft:rgba(248,113,113,.14);
  --gray:#94a3b8;--gray-soft:rgba(148,163,184,.14);
}
*,*::before,*::after{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font-family:var(--font);font-size:14px;line-height:1.55;color:var(--text);
  background:var(--bg);padding:22px 16px 60px;-webkit-font-smoothing:antialiased}
h1,h2,h3,p{margin:0}
button,input,select{font:inherit;color:inherit}
button{cursor:pointer}
:focus-visible{outline:2px solid var(--blue);outline-offset:2px;border-radius:4px}

.app{max-width:1280px;margin:0 auto;display:flex;flex-direction:column;gap:16px}

/* ---------- Header ---------- */
.header{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:22px 26px;box-shadow:var(--shadow);
  display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap}
.header__title{font-size:24px;font-weight:800;letter-spacing:-.4px;color:var(--text)}
.header__desc{font-size:13.5px;color:var(--text-muted);margin-top:6px;max-width:640px}
.header__actions{display:flex;align-items:center;gap:10px}
.status{font-size:12px;font-weight:600;padding:7px 13px;border-radius:999px;
  background:var(--surface-3);color:var(--text-muted);border:1px solid var(--border);
  display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.status::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.status.is-online{background:var(--green-soft);color:var(--green);border-color:var(--green)}
.status.is-offline{background:var(--red-soft);color:var(--red);border-color:var(--red)}
.status.is-checking{background:var(--gray-soft);color:var(--gray)}
.theme-btn{width:38px;height:38px;display:grid;place-items:center;background:var(--surface-3);
  border:1px solid var(--border);border-radius:10px;font-size:16px;transition:all var(--transition)}
.theme-btn:hover{border-color:var(--blue);color:var(--blue)}

/* ---------- Tabs ---------- */
.tabs{display:flex;gap:4px;flex-wrap:wrap;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:6px;box-shadow:var(--shadow-sm)}
.tab{display:inline-flex;align-items:center;gap:6px;padding:9px 14px;font-size:12.5px;font-weight:600;
  border-radius:var(--radius);border:none;background:transparent;color:var(--text-muted);
  transition:all var(--transition);white-space:nowrap}
.tab:hover{background:var(--surface-3);color:var(--text)}
.tab.is-active{background:var(--blue);color:#fff;box-shadow:0 4px 10px -4px rgba(37,99,235,.5)}

/* ---------- Panels ---------- */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);
  box-shadow:var(--shadow);overflow:hidden}
.panel__head{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:14px 20px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.panel__head h2{font-size:15px;font-weight:700;display:flex;align-items:center;gap:8px}
.panel__head h2::before{content:"";width:4px;height:16px;border-radius:2px;background:var(--blue)}
.panel__body{padding:18px 20px}
.tabpanel{display:none;flex-direction:column;gap:16px}
.tabpanel.is-active{display:flex}

/* ---------- Stat cards ---------- */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}
.stat{padding:16px 18px;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--surface);transition:all var(--transition)}
.stat:hover{box-shadow:var(--shadow-sm);transform:translateY(-1px)}
.stat__icon{font-size:22px;margin-bottom:6px}
.stat__label{font-size:11px;font-weight:600;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:.06em}
.stat__value{font-size:26px;font-weight:800;font-family:var(--mono);margin-top:4px;
  letter-spacing:-.5px;color:var(--text)}
.stat__value.blue{color:var(--blue)}
.stat__value.green{color:var(--green)}
.stat__value.red{color:var(--red)}
.stat__value.gray{color:var(--gray-dark)}

/* ---------- Chart cards ---------- */
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.chart-grid.two{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.chart-card{border:1px solid var(--border);border-radius:var(--radius);padding:16px;
  background:var(--surface);display:flex;flex-direction:column;gap:6px}
.chart-card__title{font-size:13px;font-weight:700;color:var(--text)}
.chart-card__desc{font-size:11.5px;color:var(--text-muted);line-height:1.5;margin-bottom:4px}
.chart-wrap{position:relative;height:260px}
.chart-wrap.tall{height:360px}

/* ---------- Table ---------- */
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface)}
table.dt{width:100%;border-collapse:collapse;font-size:12.5px;min-width:800px}
table.dt th,table.dt td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
table.dt th{background:var(--surface-3);font-weight:700;font-size:11px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--text-muted);position:sticky;top:0;cursor:pointer}
table.dt th:hover{color:var(--blue)}
table.dt tbody tr:hover{background:var(--surface-2)}
table.dt td.mono{font-family:var(--mono)}

/* ---------- Filter ---------- */
.filter-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.filter-field label{display:block;font-size:11px;font-weight:600;color:var(--text-muted);
  margin-bottom:5px;letter-spacing:.02em}
.filter-field input,.filter-field select{width:100%;padding:8px 10px;font-size:12.5px;
  background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text)}
.filter-field input:focus,.filter-field select:focus{outline:none;border-color:var(--blue);
  box-shadow:0 0 0 3px rgba(37,99,235,.18)}
.range-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.chip-group{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.chip{padding:5px 10px;font-size:11.5px;font-weight:600;border-radius:999px;
  border:1px solid var(--border);background:var(--surface);color:var(--text-muted);
  transition:all var(--transition)}
.chip.is-on{background:var(--blue-soft);color:var(--blue);border-color:var(--blue)}

/* ---------- Buttons ---------- */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:9px 18px;
  font-size:13px;font-weight:600;border-radius:10px;border:1px solid transparent;
  transition:all var(--transition)}
.btn:active:not(:disabled){transform:translateY(1px)}
.btn:disabled{opacity:.55;cursor:not-allowed}
.btn--primary{background:var(--blue);color:#fff}
.btn--primary:hover:not(:disabled){background:var(--blue-dark)}
.btn--ghost{background:var(--surface);color:var(--text-muted);border-color:var(--border)}
.btn--ghost:hover:not(:disabled){border-color:var(--blue);color:var(--blue)}
.btn--sm{padding:7px 14px;font-size:12px}

/* ---------- Pagination ---------- */
.pagination{display:flex;align-items:center;justify-content:space-between;gap:10px;
  margin-top:12px;flex-wrap:wrap}
.pagination__info{font-size:12px;color:var(--text-muted)}
.pagination__btns{display:flex;gap:6px}
.pg-btn{padding:6px 12px;font-size:12px;font-weight:600;border-radius:8px;
  border:1px solid var(--border);background:var(--surface);color:var(--text-muted)}
.pg-btn:hover:not(:disabled){border-color:var(--blue);color:var(--blue)}
.pg-btn:disabled{opacity:.4;cursor:not-allowed}

/* ---------- Prediction wizard ---------- */
.wizard{display:flex;flex-direction:column;gap:18px}
.steps{display:flex;justify-content:center;gap:12px;flex-wrap:wrap}
.step{display:flex;align-items:center;gap:10px;padding:10px 16px;border-radius:999px;
  border:1px solid var(--border);background:var(--surface-2);font-size:12.5px;
  font-weight:600;color:var(--text-muted);transition:all var(--transition)}
.step__num{width:24px;height:24px;border-radius:50%;display:grid;place-items:center;
  background:var(--surface-3);font-size:11px;font-weight:700;color:var(--text-muted)}
.step.is-active{border-color:var(--blue);color:var(--blue);background:var(--blue-soft)}
.step.is-active .step__num{background:var(--blue);color:#fff}
.step.is-done{border-color:var(--green);color:var(--green);background:var(--green-soft)}
.step.is-done .step__num{background:var(--green);color:#fff}
.wizard__body{padding:8px 4px}
.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.form-grid .field label{display:block;font-size:11.5px;font-weight:600;color:var(--text-muted);
  margin-bottom:6px}
.form-grid .field input,.form-grid .field select{width:100%;padding:10px 12px;font-size:13.5px;
  background:var(--surface);border:1px solid var(--border);border-radius:10px;color:var(--text)}
.form-grid .field input:focus,.form-grid .field select:focus{outline:none;border-color:var(--blue);
  box-shadow:0 0 0 3px rgba(37,99,235,.18)}
.wizard__nav{display:flex;justify-content:space-between;gap:10px;padding-top:12px;
  border-top:1px solid var(--border);margin-top:16px}

.predict-result{border-radius:var(--radius-lg);padding:24px;text-align:center;
  animation:fadeUp .4s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.predict-result--alive{background:var(--green-soft);border:1px solid var(--green)}
.predict-result--dead{background:var(--red-soft);border:1px solid var(--red)}
.predict-result__icon{font-size:52px;line-height:1;margin-bottom:10px}
.predict-result__label{font-size:26px;font-weight:800;letter-spacing:.05em;
  text-transform:uppercase}
.predict-result--alive .predict-result__label{color:var(--green-dark)}
.predict-result--dead .predict-result__label{color:var(--red-dark)}
.predict-result__prob{font-size:14px;margin-top:8px;color:var(--text-muted)}
.predict-result__prob strong{font-size:20px;font-family:var(--mono);
  color:var(--text);margin-left:6px}
.predict-meta{margin-top:14px;padding:12px 16px;background:var(--surface-2);
  border-radius:10px;text-align:left;font-size:12px;color:var(--text-muted)}
.predict-meta strong{color:var(--text)}

/* ---------- SVM ---------- */
.groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;padding:18px 20px}
.group{border:1px solid var(--border);border-radius:var(--radius);padding:14px;background:var(--surface-2)}
.group__title{font-size:11.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
  color:var(--blue);margin-bottom:12px;display:flex;align-items:center;gap:8px}
.group__title::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--blue)}
.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(126px,1fr));gap:10px}
.field label{display:block;font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:5px}
.field input{width:100%;padding:9px 11px;font-size:13px;font-family:var(--mono);
  background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text)}
.field input:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 3px rgba(37,99,235,.18)}
.field input.is-invalid{border-color:var(--red);background:var(--red-soft)}
.actions{display:flex;gap:10px;flex-wrap:wrap;padding:16px 20px;border-top:1px solid var(--border);
  background:var(--surface-2)}
.result__empty{padding:40px 20px;text-align:center;color:var(--text-muted)}
.result__empty .icon{font-size:40px;opacity:.3;display:block;margin-bottom:12px}
.result__card{padding:20px}
.verdict{border-radius:var(--radius-lg);padding:24px 16px;text-align:center;margin-bottom:18px}
.verdict--malignant{background:var(--red-soft);border:1px solid var(--red)}
.verdict--benign{background:var(--green-soft);border:1px solid var(--green)}
.verdict__icon{font-size:40px;line-height:1;margin-bottom:10px}
.verdict__label{font-size:20px;font-weight:800;letter-spacing:.04em;text-transform:uppercase}
.verdict--malignant .verdict__label{color:var(--red-dark)}
.verdict--benign .verdict__label{color:var(--green-dark)}
.verdict__sub{font-size:12.5px;color:var(--text-muted);margin-top:4px}
.prob{margin-bottom:16px}
.prob__head{display:flex;justify-content:space-between;font-size:12px;font-weight:600;margin-bottom:6px}
.prob__track{height:8px;border-radius:999px;background:var(--surface-3);overflow:hidden}
.prob__fill{height:100%;border-radius:999px;transition:width .55s ease}
.prob__fill--red{background:var(--red)}
.prob__fill--green{background:var(--green)}
.meta__row{display:flex;justify-content:space-between;gap:12px;padding:10px 0;font-size:13px;
  border-bottom:1px dashed var(--border)}
.meta__row:last-child{border-bottom:none}
.meta__key{color:var(--text-muted)}
.meta__val{font-weight:600;font-family:var(--mono)}
.alert{margin:0;padding:14px 16px;border-radius:var(--radius);
  background:var(--red-soft);border:1px solid var(--red);color:var(--red-dark);
  font-size:13px;line-height:1.6}
.note{margin-top:14px;padding:11px 14px;font-size:11.5px;line-height:1.65;color:var(--text-muted);
  background:var(--surface-2);border-left:3px solid var(--border-strong);
  border-radius:0 8px 8px 0}
.spinner{width:14px;height:14px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;
  border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.footer{text-align:center;font-size:12px;color:var(--text-muted);padding-top:6px}
.muted{color:var(--text-muted);font-size:12.5px}
</style>
</head>

<body>
<div class="app">

  <!-- ============== HEADER ============== -->
  <header class="header">
    <div>
      <h1 class="header__title">Phân tích dữ liệu bệnh nhân ung thư vú</h1>
      <p class="header__desc">
        Khám phá đặc điểm bệnh nhân, giai đoạn bệnh và các yếu tố liên quan đến thời gian sống.
      </p>
    </div>
    <div class="header__actions">
      <span class="status" id="status">Đang kiểm tra…</span>
      <button class="theme-btn" id="btnTheme" type="button" title="Đổi sáng/tối">🌙</button>
    </div>
  </header>

  <!-- ============== TABS ============== -->
  <nav class="tabs" id="tabs">
    <button class="tab is-active" data-tab="overview">🏠 Trang Tổng quan</button>
    <button class="tab" data-tab="data">🔍 Khám phá dữ liệu</button>
    <button class="tab" data-tab="patients">👤 Đặc điểm bệnh nhân</button>
    <button class="tab" data-tab="disease">🩺 Bệnh &amp; Khối u</button>
    <button class="tab" data-tab="biology">🧬 Yếu tố sinh học</button>
    <button class="tab" data-tab="survival">⏱️ Phân tích sống còn</button>
    <button class="tab" data-tab="predict">🤖 Dự đoán</button>
    <button class="tab" data-tab="svm">🧪 SVM 30 đặc trưng</button>
  </nav>

  <!-- ============================================================
       TAB 1 — TRANG TỔNG QUAN
       ============================================================ -->
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

    <div class="chart-grid two">
      <div class="chart-card">
        <div class="chart-card__title">Tình trạng sống: Alive / Dead</div>
        <div class="chart-card__desc">
          Biểu đồ cho thấy tỉ lệ bệnh nhân còn sống và đã tử vong trong toàn bộ dữ liệu.
        </div>
        <div class="chart-wrap"><canvas id="ovStatus"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">Phân bố các giai đoạn bệnh</div>
        <div class="chart-card__desc">
          Số lượng bệnh nhân được chẩn đoán ở từng giai đoạn (6th Stage) — giúp thấy được
          bức tranh tổng thể về mức độ phát hiện bệnh.
        </div>
        <div class="chart-wrap"><canvas id="ovStage"></canvas></div>
      </div>
    </div>
  </section>

  <!-- ============================================================
       TAB 2 — KHÁM PHÁ DỮ LIỆU
       ============================================================ -->
  <section class="tabpanel" data-panel="data">
    <div class="panel">
      <div class="panel__head">
        <h2>🔍 Bộ lọc dữ liệu</h2>
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
        <h2>📋 Dữ liệu bệnh nhân</h2>
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

  <!-- ============================================================
       TAB 3 — ĐẶC ĐIỂM BỆNH NHÂN
       ============================================================ -->
  <section class="tabpanel" data-panel="patients">
    <div class="panel">
      <div class="panel__head">
        <h2>👤 Đặc điểm bệnh nhân</h2>
        <button class="btn btn--ghost btn--sm" data-refresh="patients">🔄</button>
      </div>
      <div class="panel__body"><div class="stat-grid" id="ptStats"></div></div>
    </div>

    <div class="chart-grid two">
      <div class="chart-card">
        <div class="chart-card__title">Phân bố độ tuổi</div>
        <div class="chart-card__desc">
          Histogram cho thấy độ tuổi tập trung chủ yếu ở nhóm nào; giúp nhận diện
          nhóm tuổi có nguy cơ cao nhất.
        </div>
        <div class="chart-wrap"><canvas id="ptAge"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">Chủng tộc (Race)</div>
        <div class="chart-card__desc">
          Số lượng bệnh nhân theo từng nhóm chủng tộc trong tập dữ liệu.
        </div>
        <div class="chart-wrap"><canvas id="ptRace"></canvas></div>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-card__title">Tình trạng hôn nhân</div>
      <div class="chart-card__desc">
        Phân bố bệnh nhân theo tình trạng hôn nhân (Married / Single / Divorced / Widowed / Separated).
      </div>
      <div class="chart-wrap"><canvas id="ptMarital"></canvas></div>
    </div>
  </section>

  <!-- ============================================================
       TAB 4 — BỆNH & KHỐI U
       ============================================================ -->
  <section class="tabpanel" data-panel="disease">
    <div class="panel">
      <div class="panel__head">
        <h2>🩺 Bệnh &amp; Khối u</h2>
        <button class="btn btn--ghost btn--sm" data-refresh="disease">🔄</button>
      </div>
      <div class="panel__body"><div class="stat-grid" id="disStats"></div></div>
    </div>

    <div class="chart-grid two">
      <div class="chart-card">
        <div class="chart-card__title">Giai đoạn bệnh (6th Stage)</div>
        <div class="chart-card__desc">
          Phân bố bệnh nhân theo giai đoạn bệnh — giai đoạn càng cao thì tiên lượng càng nặng.
        </div>
        <div class="chart-wrap"><canvas id="disStage"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">Kích thước khối u (mm)</div>
        <div class="chart-card__desc">
          Histogram kích thước khối u giúp nhận diện khối u thường được phát hiện ở kích cỡ nào.
        </div>
        <div class="chart-wrap"><canvas id="disSize"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">Grade (mức độ ác tính)</div>
        <div class="chart-card__desc">
          Phân bố mức độ biệt hoá tế bào — grade càng cao, tế bào càng kém biệt hoá.
        </div>
        <div class="chart-wrap"><canvas id="disGrade"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">Differentiate (biệt hoá)</div>
        <div class="chart-card__desc">
          Mức độ biệt hoá của tế bào u: từ Well → Moderately → Poorly → Undifferentiated.
        </div>
        <div class="chart-wrap"><canvas id="disDiff"></canvas></div>
      </div>
    </div>
  </section>

  <!-- ============================================================
       TAB 5 — YẾU TỐ SINH HỌC
       ============================================================ -->
  <section class="tabpanel" data-panel="biology">
    <div class="panel">
      <div class="panel__head">
        <h2>🧬 Yếu tố sinh học</h2>
        <button class="btn btn--ghost btn--sm" data-refresh="biology">🔄</button>
      </div>
      <div class="panel__body"><div class="stat-grid" id="bioStats"></div></div>
    </div>

    <div class="chart-grid two">
      <div class="chart-card">
        <div class="chart-card__title">Estrogen Status</div>
        <div class="chart-card__desc">
          Tỉ lệ bệnh nhân dương tính / âm tính với thụ thể Estrogen.
        </div>
        <div class="chart-wrap"><canvas id="bioEst"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">Progesterone Status</div>
        <div class="chart-card__desc">
          Tỉ lệ bệnh nhân dương tính / âm tính với thụ thể Progesterone.
        </div>
        <div class="chart-wrap"><canvas id="bioProg"></canvas></div>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-card__title">Phân tích hạch bạch huyết</div>
      <div class="chart-card__desc">
        Số hạch bạch huyết dương tính (regional node positive) — chỉ số quan trọng
        phản ánh mức độ lan rộng của bệnh.
      </div>
      <div class="chart-wrap"><canvas id="bioNode"></canvas></div>
    </div>
  </section>

  <!-- ============================================================
       TAB 6 — PHÂN TÍCH SỐNG CÒN
       ============================================================ -->
  <section class="tabpanel" data-panel="survival">
    <div class="panel">
      <div class="panel__head">
        <h2>⏱️ Phân tích sống còn</h2>
        <button class="btn btn--ghost btn--sm" data-refresh="survival">🔄</button>
      </div>
      <div class="panel__body"><div class="stat-grid" id="svStats"></div></div>
    </div>

    <div class="chart-grid two">
      <div class="chart-card">
        <div class="chart-card__title">Phân bố thời gian sống</div>
        <div class="chart-card__desc">
          Histogram thời gian sống (tháng) — cho biết phần lớn bệnh nhân sống được bao lâu.
        </div>
        <div class="chart-wrap"><canvas id="svHist"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">Thời gian sống theo giai đoạn bệnh</div>
        <div class="chart-card__desc">
          Biểu đồ cho thấy sự phân bố thời gian sống ở các giai đoạn bệnh khác nhau.
        </div>
        <div class="chart-wrap"><canvas id="svBox"></canvas></div>
      </div>
    </div>

    <div class="chart-grid two">
      <div class="chart-card">
        <div class="chart-card__title">Kích thước khối u vs Thời gian sống</div>
        <div class="chart-card__desc">
          Mỗi điểm là một bệnh nhân. Xu hướng giảm cho thấy khối u lớn hơn đi kèm
          thời gian sống ngắn hơn.
        </div>
        <div class="chart-wrap"><canvas id="svScatterTumor"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-card__title">Số hạch dương tính vs Thời gian sống</div>
        <div class="chart-card__desc">
          Bệnh nhân có nhiều hạch dương tính thường có thời gian sống ngắn hơn.
        </div>
        <div class="chart-wrap"><canvas id="svScatterNode"></canvas></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel__head"><h2>Ma trận tương quan</h2></div>
      <div class="panel__body">
        <p class="muted" style="margin-bottom:10px">
          Giá trị càng gần 1 (xanh) hoặc -1 (đỏ) thì mức tương quan giữa hai biến càng mạnh.
          Ô màu đậm hơn thể hiện tương quan mạnh hơn.
        </p>
        <div class="chart-card" style="padding:8px">
          <div class="chart-wrap tall"><canvas id="svCorr"></canvas></div>
        </div>
      </div>
    </div>
  </section>

  <!-- ============================================================
       TAB 7 — DỰ ĐOÁN (wizard 3 bước)
       ============================================================ -->
  <section class="tabpanel" data-panel="predict">
    <div class="panel">
      <div class="panel__head">
        <h2>🤖 Dự đoán tình trạng bệnh nhân</h2>
      </div>
      <div class="panel__body">
        <div class="wizard">
          <div class="steps" id="wizSteps">
            <div class="step is-active" data-step="1">
              <span class="step__num">1</span><span>Thông tin bệnh nhân</span>
            </div>
            <div class="step" data-step="2">
              <span class="step__num">2</span><span>Đặc điểm bệnh</span>
            </div>
            <div class="step" data-step="3">
              <span class="step__num">3</span><span>Dự đoán</span>
            </div>
          </div>

          <div class="wizard__body">
            <div class="wizard__step" data-wstep="1">
              <div class="form-grid">
                <div class="field"><label>Tuổi</label>
                  <input type="number" id="w_age" value="52" min="18" max="120"></div>
                <div class="field"><label>Kích thước khối u (mm)</label>
                  <input type="number" id="w_tumor_size" value="25" min="1" max="200"></div>
                <div class="field"><label>Estrogen</label>
                  <select id="w_estrogen_status">
                    <option>Positive</option><option>Negative</option></select></div>
                <div class="field"><label>Progesterone</label>
                  <select id="w_progesterone_status">
                    <option>Positive</option><option>Negative</option></select></div>
              </div>
            </div>

            <div class="wizard__step" data-wstep="2" hidden>
              <div class="form-grid">
                <div class="field"><label>T Stage</label>
                  <select id="w_t_stage">
                    <option>T1</option><option selected>T2</option>
                    <option>T3</option><option>T4</option></select></div>
                <div class="field"><label>N Stage</label>
                  <select id="w_n_stage">
                    <option selected>N1</option><option>N2</option><option>N3</option></select></div>
                <div class="field"><label>Grade</label>
                  <select id="w_grade">
                    <option>1</option><option selected>2</option>
                    <option>3</option><option>4</option></select></div>
                <div class="field"><label>6th Stage</label>
                  <select id="w_stage_6th">
                    <option>IIA</option><option selected>IIB</option><option>IIIA</option>
                    <option>IIIB</option><option>IIIC</option></select></div>
              </div>
            </div>

            <div class="wizard__step" data-wstep="3" hidden>
              <div id="wizResult" style="text-align:center;padding:20px 0">
                <p class="muted">Nhấn <strong>🔮 Dự đoán</strong> để xem kết quả.</p>
              </div>
            </div>
          </div>

          <div class="wizard__nav">
            <button class="btn btn--ghost" id="wizPrev" disabled>‹ Quay lại</button>
            <button class="btn btn--primary" id="wizNext">Tiếp tục ›</button>
            <button class="btn btn--primary" id="wizPredict" hidden>🔮 Dự đoán</button>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ============================================================
       TAB 8 — SVM 30 đặc trưng (gốc)
       ============================================================ -->
  <section class="tabpanel" data-panel="svm">
    <form class="panel" id="form" autocomplete="off" novalidate>
      <div class="panel__head">
        <h2>🧪 Chẩn đoán SVM · 30 đặc trưng</h2>
        <div style="display:flex;gap:6px">
          <button type="button" class="btn btn--ghost btn--sm" id="btnSampleMalignant">⚠️ Mẫu ác tính</button>
          <button type="button" class="btn btn--ghost btn--sm" id="btnSampleBenign">✅ Mẫu lành tính</button>
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
   PALETTE — chỉ 4 màu theo quy tắc
   ============================================================ */
const CLR = {
  blue:  "#2563eb",
  green: "#16a34a",
  red:   "#dc2626",
  gray:  "#94a3b8",
  blueSoft:  "rgba(37,99,235,.65)",
  greenSoft: "rgba(22,163,74,.75)",
  redSoft:   "rgba(220,38,38,.75)",
  graySoft:  "rgba(148,163,184,.7)",
};

/* ============================================================
   MODULE 1 · CONFIG (SVM gốc)
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
  const post = (url, body) => request(url, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body||{})
  });
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
    mlPredict:(p) => post("/api/ml/predict", p),
  };
})();

/* ============================================================
   MODULE 3 · CHART REGISTRY
   ============================================================ */
const Charts = (() => {
  const registry = {};
  const css = (name) => getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
  const gridColor = () => css("--border") || "#e2e8f0";
  const textColor = () => css("--text-muted") || "#64748b";

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
  return {make, destroy, baseOpts, gridColor, textColor};
})();

/* ============================================================
   MODULE 4 · UI (SVM gốc)
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
      const label = (FEATURE_GROUPS.flatMap(g => g.fields)
        .find(([n]) => n === firstInvalid.name) || [firstInvalid.name, firstInvalid.name])[1];
      throw new Error(`Trường "${label}" (${firstInvalid.name}) ${firstInvalid.reason}.`);
    }
    return payload;
  }

  function fillSample(kind) {
    const data = SAMPLES[kind]; if (!data) return;
    for (const [k,v] of Object.entries(data)) {
      const el = $(k); if (el) { el.value = v; el.classList.remove("is-invalid"); }
    }
  }

  function clearResult() {
    $("resultEmpty").hidden = false;
    $("resultBody").hidden = true; $("resultBody").innerHTML = "";
    $("resultError").hidden = true; $("resultError").innerHTML = "";
  }

  function showError(message) {
    $("resultEmpty").hidden = true; $("resultBody").hidden = true;
    const box = $("resultError"); box.hidden = false;
    box.innerHTML = `<div class="alert" style="margin:20px"><strong>❌ Không thể phân tích</strong><br>${message}</div>`;
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
          <div class="prob__track"><div class="prob__fill prob__fill--red" style="width:${pMal}%"></div></div></div>
        <div class="prob"><div class="prob__head"><span>✅ Lành tính</span><span>${pBen.toFixed(2)}%</span></div>
          <div class="prob__track"><div class="prob__fill prob__fill--green" style="width:${pBen}%"></div></div></div>`;
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
    // Re-render charts khi theme đổi
    Charts.destroy && Object.keys(Charts).forEach(() => {});
    setTimeout(() => {
      Object.keys(ChartsRegistry).forEach(id => {
        if (ChartsRegistry[id]) { ChartsRegistry[id].options && (ChartsRegistry[id].options.plugins.legend.labels.color = Charts.textColor()); ChartsRegistry[id].update(); }
      });
    }, 50);
  }

  return { buildForm, readForm, fillSample, clearResult, showError, showResult, setLoading, setStatus, applyTheme };
})();

const ChartsRegistry = new Proxy({}, { get:(t,k)=>Charts[k] });

/* ============================================================
   MODULE 5 · TABS
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
   MODULE 6 · DASHBOARD
   ============================================================ */
const Dashboard = (() => {
  const $ = (id) => document.getElementById(id);
  const state = { meta:null, filters:{}, page:1, pageSize:15, sortBy:null, sortDir:"asc" };

  const num = (v, digits=2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" :
    (typeof v === "number" ? v.toFixed(digits) : v);

  function statCard(icon, label, value, cls="") {
    return `<div class="stat">
      <div class="stat__icon">${icon}</div>
      <div class="stat__label">${label}</div>
      <div class="stat__value ${cls}">${value}</div>
    </div>`;
  }

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

  /* ------- 1. TỔNG QUAN ------- */
  async function loadOverview() {
    try {
      const d = await Api.overview();

      $("ovStats").innerHTML =
        statCard("👥", "Tổng bệnh nhân", d.rows, "blue") +
        statCard("🟢", "Alive", d.alive, "green") +
        statCard("🔴", "Dead", d.dead, "red") +
        statCard("📈", "Thời gian sống TB", num(d.avg_survival, 1) + " th", "gray");

      // Chart 1: Alive/Dead (doughnut)
      Charts.make("ovStatus", {
        type:"doughnut",
        data:{
          labels:["Alive","Dead"],
          datasets:[{
            data:[d.alive, d.dead],
            backgroundColor:[CLR.green, CLR.red],
            borderWidth:0,
          }]
        },
        options:{
          responsive:true, maintainAspectRatio:false, cutout:"62%",
          plugins:{
            legend:{position:"bottom", labels:{color:Charts.textColor(), padding:14}},
            tooltip:{callbacks:{
              label:(c) => {
                const total = d.alive + d.dead;
                const pct = total ? ((c.raw/total)*100).toFixed(1) : "0";
                return `${c.label}: ${c.raw} (${pct}%)`;
              }
            }}
          }
        }
      });

      // Chart 2: Stage distribution (bar)
      const stageKeys = Object.keys(d.stage_distribution || {}).sort();
      Charts.make("ovStage", {
        type:"bar",
        data:{
          labels: stageKeys,
          datasets:[{
            label:"Số bệnh nhân",
            data: stageKeys.map(k => d.stage_distribution[k]),
            backgroundColor: CLR.blue,
            borderRadius: 6,
          }]
        },
        options: Charts.baseOpts()
      });
    } catch(e) {
      $("ovStats").innerHTML = `<div class="alert">${e.message}</div>`;
    }
  }

  /* ------- 2. KHÁM PHÁ DỮ LIỆU ------- */
  async function loadMeta() {
    if (state.meta) return state.meta;
    state.meta = await Api.meta();
    return state.meta;
  }

  function buildFilterUI() {
    const m = state.meta; if (!m) return;

    // Chỉ dùng 4 bộ lọc chính:
    //   Age (range), 6th Stage, Estrogen Status, Status
    const html = [];

    const ageR = m.numeric_ranges.age || {min:0, max:100};
    html.push(`<div class="filter-field">
      <label>Tuổi (${ageR.min} – ${ageR.max})</label>
      <div class="range-row">
        <input type="number" id="f_age_min" placeholder="min" value="${ageR.min}">
        <input type="number" id="f_age_max" placeholder="max" value="${ageR.max}">
      </div>
    </div>`);

    const stageVals = m.categories.stage_6th || [];
    html.push(`<div class="filter-field">
      <label>Giai đoạn (6th Stage)</label>
      <div class="chip-group" data-cat="stage_6th">
        ${stageVals.map(v => `<button type="button" class="chip" data-val="${v}">${v}</button>`).join("")}
      </div>
    </div>`);

    const estVals = m.categories.estrogen_status || [];
    html.push(`<div class="filter-field">
      <label>Estrogen Status</label>
      <div class="chip-group" data-cat="estrogen_status">
        ${estVals.map(v => `<button type="button" class="chip" data-val="${v}">${v}</button>`).join("")}
      </div>
    </div>`);

    const statusVals = m.categories.status || [];
    html.push(`<div class="filter-field">
      <label>Status</label>
      <div class="chip-group" data-cat="status">
        ${statusVals.map(v => `<button type="button" class="chip" data-val="${v}">${v}</button>`).join("")}
      </div>
    </div>`);

    $("filterGrid").innerHTML = html.join("");

    document.querySelectorAll(".chip-group").forEach(g => {
      g.addEventListener("click", (ev) => {
        const chip = ev.target.closest(".chip"); if (!chip) return;
        chip.classList.toggle("is-on");
      });
    });
  }

  function collectFilters() {
    const f = {};
    const ageLo = $("f_age_min"), ageHi = $("f_age_max");
    if (ageLo && ageLo.value !== "") f["age_min"] = Number(ageLo.value);
    if (ageHi && ageHi.value !== "") f["age_max"] = Number(ageHi.value);
    document.querySelectorAll(".chip-group").forEach(g => {
      const col = g.dataset.cat;
      const picked = Array.from(g.querySelectorAll(".chip.is-on")).map(c => c.dataset.val);
      if (picked.length) f[col] = picked;
    });
    return f;
  }

  async function loadDataTable() {
    try {
      await loadMeta();
      const payload = {filters: state.filters, page: state.page, page_size: state.pageSize,
        sort_by: state.sortBy, sort_dir: state.sortDir};
      const d = await Api.data(payload);
      $("dataCount").textContent = `${d.total} dòng · trang ${d.page}/${d.pages}`;
      // Chỉ hiển thị 6 cột quan trọng cho bảng
      const preferred = ["age","race","stage_6th","tumor_size","estrogen_status","status"];
      const cols = preferred.filter(c => d.columns.includes(c));
      const show = cols.length ? cols : d.columns.slice(0, 6);
      buildTable("dataTable", show, d.rows);

      const tbl = $("dataTable").querySelector("table");
      if (tbl) tbl.querySelectorAll("th").forEach((th, idx) => {
        th.addEventListener("click", () => {
          const col = show[idx];
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

  /* ------- 3. ĐẶC ĐIỂM BỆNH NHÂN ------- */
  async function loadPatients() {
    try {
      const d = await Api.patients({filters: state.filters});
      $("ptStats").innerHTML =
        statCard("👥","Tổng bệnh nhân", d.total, "blue") +
        statCard("📊","Tuổi TB", num(d.age_stats.mean, 1), "gray") +
        statCard("⬇️","Tuổi thấp nhất", num(d.age_stats.min, 0), "gray") +
        statCard("⬆️","Tuổi cao nhất", num(d.age_stats.max, 0), "gray");

      Charts.make("ptAge", {
        type:"bar",
        data:{labels:Object.keys(d.age_hist),
          datasets:[{label:"Số bệnh nhân", data:Object.values(d.age_hist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });

      Charts.make("ptRace", {
        type:"bar",
        data:{labels:Object.keys(d.race_dist),
          datasets:[{label:"Số bệnh nhân", data:Object.values(d.race_dist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });

      Charts.make("ptMarital", {
        type:"bar",
        data:{labels:Object.keys(d.marital_dist),
          datasets:[{label:"Số bệnh nhân", data:Object.values(d.marital_dist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:{...Charts.baseOpts(), indexAxis:"y"}
      });
    } catch(e) { $("ptStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ------- 4. BỆNH & KHỐI U ------- */
  async function loadDisease() {
    try {
      const d = await Api.disease({filters: state.filters});
      $("disStats").innerHTML =
        statCard("🩺","Tổng", d.total, "blue") +
        statCard("📏","Size TB (mm)", num(d.size_stats.mean, 1), "gray") +
        statCard("📐","Trung vị", num(d.size_stats.median, 1), "gray") +
        statCard("⬇️","Min", num(d.size_stats.min, 0), "gray") +
        statCard("⬆️","Max", num(d.size_stats.max, 0), "gray");

      Charts.make("disStage", {
        type:"bar",
        data:{labels:Object.keys(d.stage_dist),
          datasets:[{label:"Số ca", data:Object.values(d.stage_dist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });
      Charts.make("disSize", {
        type:"bar",
        data:{labels:Object.keys(d.size_hist),
          datasets:[{label:"Số ca", data:Object.values(d.size_hist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });
      Charts.make("disGrade", {
        type:"bar",
        data:{labels:Object.keys(d.grade_dist),
          datasets:[{label:"Số ca", data:Object.values(d.grade_dist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });
      Charts.make("disDiff", {
        type:"bar",
        data:{labels:Object.keys(d.differentiate_dist),
          datasets:[{label:"Số ca", data:Object.values(d.differentiate_dist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:{...Charts.baseOpts(), indexAxis:"y"}
      });
    } catch(e) { $("disStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ------- 5. SINH HỌC ------- */
  async function loadBiology() {
    try {
      const d = await Api.biology({filters: state.filters});
      $("bioStats").innerHTML =
        statCard("🧬","Tổng", d.total, "blue") +
        statCard("🟢","Hạch (+) TB", num(d.node_stats.mean_positive, 2), "red") +
        statCard("🔵","Hạch kiểm tra TB", num(d.node_stats.mean_examined, 1), "gray") +
        statCard("⚠️","Max hạch (+)", num(d.node_stats.max_positive, 0), "red");

      const estLabels = Object.keys(d.estrogen_dist);
      const estColors = estLabels.map(l => l.toLowerCase() === "positive" ? CLR.green : CLR.red);
      Charts.make("bioEst", {
        type:"doughnut",
        data:{labels:estLabels, datasets:[{data:Object.values(d.estrogen_dist),
          backgroundColor:estColors, borderWidth:0}]},
        options:{responsive:true, maintainAspectRatio:false, cutout:"62%",
          plugins:{legend:{position:"bottom", labels:{color:Charts.textColor()}}}}
      });

      const progLabels = Object.keys(d.progesterone_dist);
      const progColors = progLabels.map(l => l.toLowerCase() === "positive" ? CLR.green : CLR.red);
      Charts.make("bioProg", {
        type:"doughnut",
        data:{labels:progLabels, datasets:[{data:Object.values(d.progesterone_dist),
          backgroundColor:progColors, borderWidth:0}]},
        options:{responsive:true, maintainAspectRatio:false, cutout:"62%",
          plugins:{legend:{position:"bottom", labels:{color:Charts.textColor()}}}}
      });

      Charts.make("bioNode", {
        type:"bar",
        data:{labels:Object.keys(d.node_hist),
          datasets:[{label:"Số ca", data:Object.values(d.node_hist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });
    } catch(e) { $("bioStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  /* ------- 6. SỐNG CÒN ------- */
  async function loadSurvival() {
    try {
      const d = await Api.survival({filters: state.filters});
      $("svStats").innerHTML =
        statCard("⏱️","Tổng", d.total, "blue") +
        statCard("📈","Tháng TB", num(d.survival_stats.mean, 1), "gray") +
        statCard("📊","Trung vị", num(d.survival_stats.median, 1), "gray") +
        statCard("⬇️","Min", num(d.survival_stats.min, 0), "gray") +
        statCard("⬆️","Max", num(d.survival_stats.max, 0), "gray");

      // Histogram
      Charts.make("svHist", {
        type:"bar",
        data:{labels:Object.keys(d.survival_hist),
          datasets:[{label:"Số ca", data:Object.values(d.survival_hist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });

      // Boxplot theo stage (dùng bar với floating [q1, q3])
      const stages = Object.keys(d.survival_boxplot_by_stage || {});
      const boxes = stages.map(s => {
        const b = d.survival_boxplot_by_stage[s];
        return {min:b.min, q1:b.q1, median:b.median, q3:b.q3, max:b.max};
      });

      if (window.Chart && stages.length) {
        Charts.make("svBox", {
          type:"boxplot",
          data:{
            labels: stages,
            datasets:[{
              label:"Thời gian sống (tháng)",
              data: boxes,
              backgroundColor: CLR.blueSoft,
              borderColor: CLR.blue,
              borderWidth: 1.5,
              medianColor: CLR.blue,
              itemRadius: 0,
              outlierRadius: 2,
              outlierBackgroundColor: CLR.red,
            }]
          },
          options:{
            responsive:true, maintainAspectRatio:false,
            plugins:{legend:{display:false}},
            scales:{
              x:{ticks:{color:Charts.textColor()}, grid:{color:Charts.gridColor()}},
              y:{ticks:{color:Charts.textColor()}, grid:{color:Charts.gridColor()}, beginAtZero:true}
            }
          }
        });
      }

      // Scatter: Tumor size vs survival
      const scT = d.scatter_tumor_survival || [];
      const aliveT = scT.filter(p => p.status === "Alive");
      const deadT  = scT.filter(p => p.status === "Dead");
      Charts.make("svScatterTumor", {
        type:"scatter",
        data:{datasets:[
          {label:"Alive", data:aliveT.map(p => ({x:p.x, y:p.y})),
           backgroundColor:CLR.greenSoft, pointRadius:3},
          {label:"Dead", data:deadT.map(p => ({x:p.x, y:p.y})),
           backgroundColor:CLR.redSoft, pointRadius:3},
        ]},
        options:{...Charts.baseOpts(), plugins:{
          legend:{labels:{color:Charts.textColor()}}
        }}
      });

      const scN = d.scatter_node_survival || [];
      const aliveN = scN.filter(p => p.status === "Alive");
      const deadN  = scN.filter(p => p.status === "Dead");
      Charts.make("svScatterNode", {
        type:"scatter",
        data:{datasets:[
          {label:"Alive", data:aliveN.map(p => ({x:p.x, y:p.y})),
           backgroundColor:CLR.greenSoft, pointRadius:3},
          {label:"Dead", data:deadN.map(p => ({x:p.x, y:p.y})),
           backgroundColor:CLR.redSoft, pointRadius:3},
        ]},
        options:{...Charts.baseOpts()}
      });

      // Correlation matrix
      const cols = d.correlation_columns || [];
      if (cols.length && typeof Chart.registry.getController("matrix") !== "undefined") {
        const pts = [];
        cols.forEach((row) => {
          cols.forEach((col) => {
            pts.push({x:col, y:row, v:d.correlation[row]?.[col] ?? 0});
          });
        });
        Charts.destroy("svCorr");
        const el = document.getElementById("svCorr");
        if (el) {
          new Chart(el.getContext("2d"), {
            type:"matrix",
            data:{datasets:[{
              data:pts,
              backgroundColor:(c) => {
                const v = c.raw?.v ?? 0; const a = Math.min(1, Math.abs(v));
                return v >= 0 ? `rgba(37,99,235,${0.15 + 0.75*a})`
                              : `rgba(220,38,38,${0.15 + 0.75*a})`;
              },
              width:({chart}) => ((chart.chartArea||{width:400}).width / cols.length) - 2,
              height:({chart}) => ((chart.chartArea||{height:400}).height / cols.length) - 2,
            }]},
            options:{
              responsive:true, maintainAspectRatio:false,
              plugins:{
                legend:{display:false},
                tooltip:{callbacks:{
                  title:()=> "",
                  label:(c) => `${c.raw.y} × ${c.raw.x}: ${c.raw.v}`
                }}
              },
              scales:{
                x:{type:"category", labels:cols,
                   ticks:{color:Charts.textColor(), font:{size:9}}},
                y:{type:"category", labels:cols,
                   ticks:{color:Charts.textColor(), font:{size:9}}}
              }
            }
          });
        }
      } else if (el_placeholder_guard()) {}
    } catch(e) {
      $("svStats").innerHTML = `<div class="alert">${e.message}</div>`;
    }
  }
  function el_placeholder_guard(){ return false; }

  /* ------- ML dùng cho tab Dự đoán ------- */
  async function ensureModel() {
    try {
      const d = await Api.train();
      return d;
    } catch(e) { return null; }
  }

  /* ------- LOAD DISPATCH ------- */
  async function load(name) {
    try {
      if (name === "overview") { await loadOverview(); }
      else if (name === "data") { await loadMeta(); buildFilterUI(); await loadDataTable(); }
      else if (name === "patients") { await loadPatients(); }
      else if (name === "disease") { await loadDisease(); }
      else if (name === "biology") { await loadBiology(); }
      else if (name === "survival") { await loadSurvival(); }
      else if (name === "predict") { await ensureModel(); }
    } catch(e) { console.error("[load]", name, e); }
  }

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
  }

  return { load, bind, state };
})();

/* ============================================================
   MODULE 6b · PREDICTION WIZARD (3 bước)
   ============================================================ */
const Wizard = (() => {
  const $ = (id) => document.getElementById(id);
  let step = 1;

  function showStep(n) {
    step = n;
    document.querySelectorAll(".wizard__step").forEach(el => {
      el.hidden = Number(el.dataset.wstep) !== n;
    });
    document.querySelectorAll(".step").forEach(el => {
      const s = Number(el.dataset.step);
      el.classList.toggle("is-active", s === n);
      el.classList.toggle("is-done", s < n);
    });
    $("wizPrev").disabled = n <= 1;
    $("wizNext").hidden = n >= 3;
    $("wizPredict").hidden = n !== 3;
  }

  function collectAll() {
    const v = id => $(id)?.value;
    return {
      age: Number(v("w_age")) || 50,
      tumor_size: Number(v("w_tumor_size")) || 25,
      estrogen_status: v("w_estrogen_status") || "Positive",
      progesterone_status: v("w_progesterone_status") || "Positive",
      t_stage: v("w_t_stage") || "T2",
      n_stage: v("w_n_stage") || "N1",
      grade: Number(v("w_grade")) || 2,
      stage_6th: v("w_stage_6th") || "IIB",
      // Các trường còn lại dùng giá trị mặc định hợp lý
      race: "White",
      marital_status: "Married",
      differentiate: "Moderately differentiated",
      a_stage: "Regional",
      regional_node_examined: 10,
      regional_node_positive: 0,
      survival_months: 60,
      model: "Random Forest",
    };
  }

  async function predict() {
    const result = $("wizResult");
    result.innerHTML = `<p class="muted">⏳ Đang dự đoán…</p>`;
    try {
      const payload = collectAll();
      const d = await Api.mlPredict(payload);
      const isAlive = d.prediction === "Alive";
      const cls = isAlive ? "alive" : "dead";
      const icon = isAlive ? "🟢" : "🔴";
      const mainProb = isAlive ? d.probability_alive : d.probability_dead;
      result.innerHTML = `
        <div class="predict-result predict-result--${cls}">
          <div class="predict-result__icon">${icon}</div>
          <div class="predict-result__label">${d.prediction}</div>
          <div class="predict-result__prob">Xác suất: <strong>${(mainProb*100).toFixed(1)}%</strong></div>
        </div>
        <div class="predict-meta">
          <p>Mô hình sử dụng: <strong>${d.model}</strong></p>
          <p>P(Alive) = <strong>${(d.probability_alive*100).toFixed(1)}%</strong> ·
             P(Dead) = <strong>${(d.probability_dead*100).toFixed(1)}%</strong></p>
          <p style="margin-top:6px;color:var(--text-soft)">
            Kết quả chỉ mang tính tham khảo, không thay thế kết luận của bác sĩ.
          </p>
        </div>`;
    } catch(e) {
      result.innerHTML = `<div class="alert">${e.message}</div>`;
    }
  }

  function init() {
    $("wizPrev").addEventListener("click", () => showStep(Math.max(1, step - 1)));
    $("wizNext").addEventListener("click", () => showStep(Math.min(3, step + 1)));
    $("wizPredict").addEventListener("click", predict);
    showStep(1);
  }
  return { init };
})();

/* ============================================================
   MODULE 7 · APP (SVM + khởi động)
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
    Wizard.init();
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
