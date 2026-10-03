from fastapi import FastAPI, HTTPException, Body, Query, Request, Response, Depends, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import pandas as pd
import io
import math
import numpy as np
import joblib
import uvicorn

# ==== SECURITY (SQL) ====
import sqlite3
import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

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
# 0. SECURITY LAYER — SQLite auth, session, audit, rate-limit
# ==============================================================
DB_PATH             = os.environ.get("BC_DB_PATH", "auth.db")
SESSION_COOKIE      = "bc_session"
SESSION_TTL_HOURS   = int(os.environ.get("BC_SESSION_TTL_HOURS", "24"))
MAX_FAILED_LOGINS   = int(os.environ.get("BC_MAX_FAILED", "5"))
LOCKOUT_MINUTES     = int(os.environ.get("BC_LOCKOUT_MINUTES", "15"))
PBKDF2_ITERATIONS   = 200_000
DEFAULT_ADMIN_USER  = os.environ.get("BC_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASS  = os.environ.get("BC_ADMIN_PASS", "Admin@123")
ALLOW_REGISTRATION  = os.environ.get("BC_ALLOW_REGISTER", "1") == "1"

# ---- SQL schema (parameterized queries everywhere → chống SQL Injection)
SQL_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash  TEXT    NOT NULL,
    full_name      TEXT    DEFAULT '',
    role           TEXT    NOT NULL DEFAULT 'analyst',
    is_active      INTEGER NOT NULL DEFAULT 1,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until   TEXT,
    created_at     TEXT    NOT NULL,
    last_login_at  TEXT
);
"""

SQL_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    ip          TEXT,
    user_agent  TEXT,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sessions_user   ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
"""

SQL_CREATE_AUDIT = """
CREATE TABLE IF NOT EXISTS login_audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT,
    ip         TEXT,
    user_agent TEXT,
    success    INTEGER NOT NULL,
    reason     TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_user ON login_audit(username);
CREATE INDEX IF NOT EXISTS idx_audit_time ON login_audit(created_at);
"""

SQL_CREATE_USERS_TS = """
CREATE TRIGGER IF NOT EXISTS trg_users_updated
AFTER UPDATE OF password_hash ON users
BEGIN
    UPDATE users SET last_login_at = last_login_at WHERE id = NEW.id;
END;
"""


@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_auth_db() -> None:
    """Khởi tạo schema + seed admin mặc định."""
    with db_conn() as conn:
        conn.executescript(SQL_CREATE_USERS)
        conn.executescript(SQL_CREATE_SESSIONS)
        conn.executescript(SQL_CREATE_AUDIT)

        # Seed default admin nếu DB trống
        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
        if row["c"] == 0:
            conn.execute(
                "INSERT INTO users (username, password_hash, full_name, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (DEFAULT_ADMIN_USER, hash_password(DEFAULT_ADMIN_PASS),
                 "Administrator", "admin", _now_iso()),
            )
            print(f"[AUTH] Seeded default admin user: {DEFAULT_ADMIN_USER}")


# ---------- Password hashing (PBKDF2-HMAC-SHA256) ----------
def hash_password(password: str) -> str:
    if not password or len(password) < 6:
        raise ValueError("Mật khẩu phải có ít nhất 6 ký tự.")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return secrets.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# ---------- Audit log ----------
def audit_log(username: Optional[str], ip: Optional[str], ua: Optional[str],
              success: bool, reason: str = "") -> None:
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO login_audit (username, ip, user_agent, success, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username or "", ip or "", (ua or "")[:300], 1 if success else 0,
                 reason[:200], _now_iso()),
            )
    except Exception as exc:
        print(f"[AUDIT] {exc}")


# ---------- Authenticate / create session ----------
def authenticate(username: str, password: str, ip: str, ua: str) -> Dict[str, Any]:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, full_name, role, is_active, "
            "       failed_attempts, locked_until "
            "FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()

        if row is None:
            audit_log(username, ip, ua, False, "user_not_found")
            return {"ok": False, "reason": "Tên đăng nhập hoặc mật khẩu không đúng."}

        if not row["is_active"]:
            audit_log(username, ip, ua, False, "inactive")
            return {"ok": False, "reason": "Tài khoản đã bị vô hiệu hoá."}

        # Lock check
        locked_until = row["locked_until"]
        if locked_until:
            try:
                lu = datetime.fromisoformat(locked_until)
                if lu.tzinfo is None:
                    lu = lu.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) < lu:
                    remaining = int((lu - datetime.now(timezone.utc)).total_seconds() // 60) + 1
                    audit_log(username, ip, ua, False, "locked")
                    return {"ok": False,
                            "reason": f"Tài khoản tạm khoá. Thử lại sau {remaining} phút."}
            except Exception:
                pass

        if not verify_password(password, row["password_hash"]):
            failed = row["failed_attempts"] + 1
            new_lock = None
            if failed >= MAX_FAILED_LOGINS:
                new_lock = (datetime.now(timezone.utc)
                            + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(timespec="seconds")
            conn.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                (failed, new_lock, row["id"]),
            )
            audit_log(username, ip, ua, False, "bad_password")
            if new_lock:
                return {"ok": False,
                        "reason": f"Sai mật khẩu quá {MAX_FAILED_LOGINS} lần. "
                                  f"Tài khoản bị khoá {LOCKOUT_MINUTES} phút."}
            return {"ok": False, "reason": "Tên đăng nhập hoặc mật khẩu không đúng."}

        # Success
        conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL, "
            "                 last_login_at = ? WHERE id = ?",
            (_now_iso(), row["id"]),
        )
        audit_log(username, ip, ua, True, "ok")
        return {
            "ok": True,
            "user": {"id": row["id"], "username": row["username"],
                     "full_name": row["full_name"], "role": row["role"]},
        }


def create_session(user_id: int, ip: str, ua: str) -> str:
    token = secrets.token_urlsafe(48)
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(hours=SESSION_TTL_HOURS)).isoformat(timespec="seconds")
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, ip, user_agent, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (token, user_id, ip or "", (ua or "")[:300], _now_iso(), expires),
        )
    return token


def get_session(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT s.token, s.expires_at, u.id AS user_id, u.username, "
            "       u.full_name, u.role, u.is_active "
            "FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        try:
            exp = datetime.fromisoformat(row["expires_at"])
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) >= exp:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                return None
        except Exception:
            return None
        if not row["is_active"]:
            return None
        return {"user_id": row["user_id"], "username": row["username"],
                "full_name": row["full_name"], "role": row["role"], "token": row["token"]}


def revoke_session(token: Optional[str]) -> None:
    if not token:
        return
    with db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def cleanup_expired_sessions() -> None:
    try:
        with db_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now_iso(),))
    except Exception:
        pass


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


class LoginPayload(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=200)


class RegisterPayload(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=200)
    full_name: str = ""


CANCER_CLASSES = {0: "malignant", 1: "benign"}
CLASS_LABELS = ["malignant", "benign"]


# ==============================================================
# 3. FASTAPI APP
# ==============================================================
app = FastAPI(
    title="Breast Cancer Analytics API",
    description="Dashboard phân tích sống còn + SVM chẩn đoán 30 đặc trưng. Có bảo mật SQL.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo DB ngay khi app start
@app.on_event("startup")
def _startup():
    init_auth_db()
    cleanup_expired_sessions()


# ---------- Middleware xác thực ----------
PUBLIC_PATHS = {
    "/login", "/api/auth/login", "/api/auth/register",
    "/favicon.ico", "/docs", "/openapi.json", "/redoc",
}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path

    if path in PUBLIC_PATHS or path.startswith("/static"):
        return await call_next(request)

    token = request.cookies.get(SESSION_COOKIE)
    session = get_session(token)
    if session is None:
        if path.startswith("/api/") or path in ("/predict", "/health"):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return RedirectResponse(url="/login", status_code=302)

    request.state.user = session
    return await call_next(request)


# ==============================================================
# 3a. AUTH ENDPOINTS
# ==============================================================
def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    # Nếu đã đăng nhập → chuyển về dashboard
    if get_session(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/", status_code=302)
    return LOGIN_TEMPLATE


@app.post("/api/auth/login")
def api_login(payload: LoginPayload, request: Request, response: Response):
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")
    result = authenticate(payload.username, payload.password, ip, ua)
    if not result["ok"]:
        raise HTTPException(status_code=401, detail=result["reason"])

    user = result["user"]
    token = create_session(user["id"], ip, ua)
    response.set_cookie(
        key=SESSION_COOKIE, value=token,
        max_age=SESSION_TTL_HOURS * 3600,
        httponly=True, samesite="lax", secure=False,  # secure=True khi chạy HTTPS
        path="/",
    )
    return {"ok": True, "user": user}


@app.post("/api/auth/register")
def api_register(payload: RegisterPayload, request: Request, response: Response):
    if not ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="Chức năng đăng ký đã bị tắt.")
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")

    # Validate sơ bộ
    if not payload.username.replace("_", "").replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="Username chỉ gồm chữ, số, '_', '.'.")

    try:
        pw_hash = hash_password(payload.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        with db_conn() as conn:
            # Kiểm tra tồn tại (parameterized → chống SQL Injection)
            exists = conn.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE",
                (payload.username,),
            ).fetchone()
            if exists:
                raise HTTPException(status_code=409, detail="Username đã tồn tại.")
            conn.execute(
                "INSERT INTO users (username, password_hash, full_name, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (payload.username, pw_hash, payload.full_name or payload.username,
                 "analyst", _now_iso()),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo tài khoản: {exc}")

    # Auto login
    result = authenticate(payload.username, payload.password, ip, ua)
    if not result["ok"]:
        return {"ok": True, "user": None, "message": "Đăng ký thành công, hãy đăng nhập."}

    user = result["user"]
    token = create_session(user["id"], ip, ua)
    response.set_cookie(
        key=SESSION_COOKIE, value=token,
        max_age=SESSION_TTL_HOURS * 3600,
        httponly=True, samesite="lax", secure=False, path="/",
    )
    return {"ok": True, "user": user}


@app.post("/api/auth/logout")
def api_logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    revoke_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def api_me(request: Request):
    session = getattr(request.state, "user", None) or get_session(request.cookies.get(SESSION_COOKIE))
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"user": {"id": session["user_id"], "username": session["username"],
                     "full_name": session["full_name"], "role": session["role"]}}


# ==============================================================
# 3b. HEALTH
# ==============================================================
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


# ==============================================================
# 3c. PREDICT (SVM)
# ==============================================================
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
# 3d. DASHBOARD APIs
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


@app.post("/api/analysis/biology")
def api_biology(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    def _dist(col):
        return {str(k): int(v) for k, v in d[col].value_counts().to_dict().items()}

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


@app.post("/api/analysis/survival")
def api_survival(payload: FilterPayload):
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng.")
    d = _apply_filters(_df, payload.filters)

    bins = list(range(0, 121, 10))
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
    surv_hist = (pd.cut(d["survival_months"], bins=bins, labels=labels, right=False)
                 .value_counts().reindex(labels).fillna(0).astype(int).to_dict())

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
# 4. FRONTEND — Dashboard
# ==============================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return HTML_TEMPLATE


# ==============================================================
# 4a. LOGIN TEMPLATE
# ==============================================================
LOGIN_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Đăng nhập · Breast Cancer Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --blue:#3b82f6; --blue-2:#2563eb; --blue-3:#1d4ed8;
  --green:#10b981; --red:#ef4444; --amber:#f59e0b;
  --bg:#060913; --surface:rgba(15,23,42,.72); --surface-2:rgba(30,41,59,.6);
  --text:#f1f5f9; --text-muted:#94a3b8; --text-soft:#64748b;
  --border:rgba(148,163,184,.18); --border-strong:rgba(148,163,184,.32);
  --radius:14px; --radius-lg:20px;
  --font:'Plus Jakarta Sans',system-ui,sans-serif;
  --mono:'JetBrains Mono',monospace;
}
[data-theme="light"]{
  --bg:#f1f5f9; --surface:rgba(255,255,255,.85); --surface-2:rgba(241,245,249,.7);
  --text:#0f172a; --text-muted:#475569; --text-soft:#94a3b8;
  --border:rgba(15,23,42,.08); --border-strong:rgba(15,23,42,.18);
}
*,*::before,*::after{box-sizing:border-box}
html,body{height:100%;margin:0}
body{
  font-family:var(--font); color:var(--text); background:var(--bg);
  overflow-x:hidden; -webkit-font-smoothing:antialiased;
  display:grid; place-items:center; min-height:100vh; padding:24px;
}

/* === Animated gradient orbs background === */
.bg{position:fixed;inset:0;overflow:hidden;z-index:0;pointer-events:none}
.orb{position:absolute;border-radius:50%;filter:blur(80px);opacity:.55;
  animation:float 22s ease-in-out infinite}
.orb.o1{width:520px;height:520px;background:radial-gradient(circle,#2563eb,#1e40af);top:-160px;left:-120px;animation-delay:0s}
.orb.o2{width:460px;height:460px;background:radial-gradient(circle,#10b981,#047857);bottom:-140px;right:-100px;animation-delay:-7s}
.orb.o3{width:420px;height:420px;background:radial-gradient(circle,#8b5cf6,#4c1d95);top:40%;left:60%;animation-delay:-14s}
@keyframes float{
  0%,100%{transform:translate(0,0) scale(1)}
  33%{transform:translate(60px,-40px) scale(1.08)}
  66%{transform:translate(-50px,50px) scale(.95)}
}
.grid-overlay{position:fixed;inset:0;z-index:1;pointer-events:none;opacity:.35;
  background-image:
    linear-gradient(rgba(148,163,184,.06) 1px,transparent 1px),
    linear-gradient(90deg,rgba(148,163,184,.06) 1px,transparent 1px);
  background-size:44px 44px;
  mask-image:radial-gradient(ellipse at center,black 40%,transparent 75%);
}

/* === Layout === */
.shell{position:relative;z-index:2;width:100%;max-width:1080px;
  display:grid;grid-template-columns:1fr 1fr;gap:0;
  background:var(--surface);backdrop-filter:blur(24px) saturate(160%);
  -webkit-backdrop-filter:blur(24px) saturate(160%);
  border:1px solid var(--border);border-radius:24px;
  box-shadow:0 30px 60px -20px rgba(0,0,0,.5), 0 0 0 1px rgba(255,255,255,.02) inset;
  overflow:hidden;animation:rise .7s cubic-bezier(.2,.9,.3,1.2)}
@keyframes rise{from{opacity:0;transform:translateY(20px) scale(.98)}to{opacity:1;transform:none}}

/* === Left brand panel === */
.brand{padding:52px 44px;background:
  linear-gradient(150deg,rgba(37,99,235,.22) 0%,rgba(16,185,129,.10) 55%,transparent 100%);
  border-right:1px solid var(--border);display:flex;flex-direction:column;gap:20px;position:relative;overflow:hidden}
.brand::before{content:"";position:absolute;inset:0;
  background:radial-gradient(circle at 20% 10%,rgba(59,130,246,.35),transparent 55%),
             radial-gradient(circle at 80% 90%,rgba(16,185,129,.28),transparent 55%);
  pointer-events:none}
.brand__logo{display:flex;align-items:center;gap:12px;position:relative;z-index:1}
.logo-mark{width:46px;height:46px;border-radius:13px;display:grid;place-items:center;
  background:linear-gradient(135deg,#3b82f6,#2563eb 45%,#1e40af);
  box-shadow:0 10px 30px -8px rgba(37,99,235,.6),0 0 0 1px rgba(255,255,255,.15) inset;
  color:#fff;font-size:22px;font-weight:800;letter-spacing:-1px}
.brand__name{font-size:16px;font-weight:800;letter-spacing:-.3px}
.brand__tag{font-size:11.5px;color:var(--text-muted);letter-spacing:.05em;text-transform:uppercase}
.brand h1{font-size:34px;font-weight:800;letter-spacing:-1px;line-height:1.15;
  margin:8px 0 0;position:relative;z-index:1;
  background:linear-gradient(120deg,var(--text),var(--text-muted) 130%);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.brand p{color:var(--text-muted);font-size:14px;line-height:1.6;margin:0;position:relative;z-index:1;max-width:380px}
.features{display:flex;flex-direction:column;gap:14px;margin-top:auto;position:relative;z-index:1}
.feat{display:flex;gap:12px;align-items:flex-start;padding:12px 14px;border-radius:12px;
  background:var(--surface-2);border:1px solid var(--border);
  backdrop-filter:blur(10px);transition:all .25s ease}
.feat:hover{transform:translateX(4px);border-color:var(--border-strong)}
.feat__ic{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;
  background:linear-gradient(135deg,rgba(59,130,246,.22),rgba(37,99,235,.12));
  color:#60a5fa;font-size:16px;flex-shrink:0;border:1px solid rgba(96,165,250,.25)}
.feat__t{font-size:13px;font-weight:700}
.feat__d{font-size:11.5px;color:var(--text-muted);margin-top:2px;line-height:1.5}

/* === Right form panel === */
.form{padding:52px 44px;display:flex;flex-direction:column;justify-content:center;gap:20px}
.form__head{margin-bottom:4px}
.form__head h2{font-size:26px;font-weight:800;letter-spacing:-.6px;margin:0}
.form__head p{color:var(--text-muted);font-size:13px;margin:6px 0 0}
.tabs{display:flex;gap:4px;padding:4px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:11px;margin-bottom:6px}
.tab{flex:1;padding:9px 12px;font-size:12.5px;font-weight:600;background:transparent;
  border:none;border-radius:8px;color:var(--text-muted);cursor:pointer;transition:all .2s}
.tab.on{background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;
  box-shadow:0 8px 20px -8px rgba(59,130,246,.7)}
.field{display:flex;flex-direction:column;gap:6px}
.field label{font-size:11.5px;font-weight:700;color:var(--text-muted);letter-spacing:.03em;
  text-transform:uppercase}
.field .wrap{position:relative}
.field .wrap::before{content:attr(data-icon);position:absolute;left:14px;top:50%;
  transform:translateY(-50%);font-size:15px;opacity:.55;pointer-events:none}
.field input{width:100%;padding:12px 14px 12px 42px;font-size:14px;font-family:var(--font);
  color:var(--text);background:var(--surface-2);
  border:1px solid var(--border-strong);border-radius:11px;transition:all .2s ease}
.field input:focus{outline:none;border-color:#3b82f6;background:rgba(59,130,246,.06);
  box-shadow:0 0 0 4px rgba(59,130,246,.18)}
.btn{width:100%;padding:13px 18px;font-size:14px;font-weight:700;color:#fff;border:none;
  border-radius:11px;cursor:pointer;position:relative;overflow:hidden;
  background:linear-gradient(135deg,#3b82f6,#2563eb 50%,#1d4ed8);
  box-shadow:0 14px 26px -12px rgba(37,99,235,.8),0 0 0 1px rgba(255,255,255,.08) inset;
  transition:all .25s ease;letter-spacing:.02em}
.btn:hover:not(:disabled){transform:translateY(-1px);
  box-shadow:0 18px 32px -12px rgba(37,99,235,.95)}
.btn:active:not(:disabled){transform:translateY(0)}
.btn:disabled{opacity:.7;cursor:not-allowed}
.btn .spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.35);
  border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;
  vertical-align:-2px;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}

.msg{padding:11px 14px;border-radius:10px;font-size:12.5px;line-height:1.55;
  display:none;align-items:flex-start;gap:8px;animation:fade .3s ease}
.msg.show{display:flex}
.msg--err{background:rgba(239,68,68,.12);color:#fca5a5;border:1px solid rgba(239,68,68,.35)}
.msg--ok{background:rgba(16,185,129,.12);color:#6ee7b7;border:1px solid rgba(16,185,129,.35)}
@keyframes fade{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}

.hint{font-size:11.5px;color:var(--text-soft);text-align:center;line-height:1.6}
.hint code{font-family:var(--mono);font-size:11px;background:var(--surface-2);
  padding:2px 6px;border-radius:5px;color:#93c5fd;border:1px solid var(--border)}
.secure{display:flex;align-items:center;justify-content:center;gap:8px;
  font-size:11px;color:var(--text-soft);letter-spacing:.03em;margin-top:6px}
.secure span{display:inline-flex;align-items:center;gap:5px;
  padding:5px 10px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border)}
.pw-toggle{position:absolute;right:12px;top:50%;transform:translateY(-50%);
  background:transparent;border:none;color:var(--text-muted);cursor:pointer;font-size:14px;padding:4px}

@media (max-width: 860px){
  .shell{grid-template-columns:1fr;max-width:480px}
  .brand{padding:34px 26px;border-right:none;border-bottom:1px solid var(--border)}
  .brand h1{font-size:26px}
  .features{display:none}
  .form{padding:34px 26px}
}
</style>
</head>
<body>
<div class="bg">
  <div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>
</div>
<div class="grid-overlay"></div>

<main class="shell">
  <!-- Brand panel -->
  <section class="brand">
    <div class="brand__logo">
      <div class="logo-mark">BC</div>
      <div>
        <div class="brand__name">Breast Cancer Analytics</div>
        <div class="brand__tag">Clinical Intelligence Suite</div>
      </div>
    </div>
    <h1>Bảng điều khiển<br>phân tích ung thư vú.</h1>
    <p>Khám phá dữ liệu sống còn, so sánh mô hình học máy và chẩn đoán bằng SVM 30 đặc trưng trong một nền tảng duy nhất.</p>
    <div class="features">
      <div class="feat">
        <div class="feat__ic">🛡️</div>
        <div>
          <div class="feat__t">Bảo mật SQL đa lớp</div>
          <div class="feat__d">PBKDF2 + salt, session token ngẫu nhiên, audit log, rate-limit.</div>
        </div>
      </div>
      <div class="feat">
        <div class="feat__ic">📊</div>
        <div>
          <div class="feat__t">Dashboard trực quan</div>
          <div class="feat__d">8 tab phân tích: bệnh nhân, khối u, sinh học, sống còn, ML…</div>
        </div>
      </div>
      <div class="feat">
        <div class="feat__ic">🧪</div>
        <div>
          <div class="feat__t">SVM + 4 mô hình ML</div>
          <div class="feat__d">Logistic, RF, Gradient Boosting, Decision Tree — so sánh AUC tức thời.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- Form panel -->
  <section class="form">
    <div class="form__head">
      <h2>Đăng nhập hệ thống</h2>
      <p>Vui lòng xác thực để truy cập dashboard.</p>
    </div>

    <div class="tabs">
      <button class="tab on" id="tabLogin" type="button">Đăng nhập</button>
      <button class="tab" id="tabRegister" type="button">Đăng ký</button>
    </div>

    <div class="msg" id="msg"></div>

    <!-- Login form -->
    <form id="formLogin" autocomplete="on">
      <div class="field">
        <label for="loginUser">Tên đăng nhập</label>
        <div class="wrap" data-icon="👤">
          <input id="loginUser" name="username" type="text" required
                 minlength="3" maxlength="64" placeholder="admin" autocomplete="username">
        </div>
      </div>
      <div class="field" style="margin-top:14px">
        <label for="loginPass">Mật khẩu</label>
        <div class="wrap" data-icon="🔒">
          <input id="loginPass" name="password" type="password" required
                 minlength="6" maxlength="200" placeholder="••••••••" autocomplete="current-password">
          <button type="button" class="pw-toggle" id="togglePw1">👁️</button>
        </div>
      </div>
      <button class="btn" id="btnLogin" type="submit" style="margin-top:22px">
        <span id="loginTxt">Đăng nhập</span>
      </button>
    </form>

    <!-- Register form -->
    <form id="formRegister" hidden autocomplete="off">
      <div class="field">
        <label for="regUser">Tên đăng nhập</label>
        <div class="wrap" data-icon="👤">
          <input id="regUser" name="username" type="text" required
                 minlength="3" maxlength="64" placeholder="analyst01">
        </div>
      </div>
      <div class="field" style="margin-top:14px">
        <label for="regName">Họ và tên</label>
        <div class="wrap" data-icon="🪪">
          <input id="regName" name="full_name" type="text" maxlength="120" placeholder="Nguyễn Văn A">
        </div>
      </div>
      <div class="field" style="margin-top:14px">
        <label for="regPass">Mật khẩu</label>
        <div class="wrap" data-icon="🔒">
          <input id="regPass" name="password" type="password" required
                 minlength="6" maxlength="200" placeholder="Ít nhất 6 ký tự">
          <button type="button" class="pw-toggle" id="togglePw2">👁️</button>
        </div>
      </div>
      <button class="btn" id="btnRegister" type="submit" style="margin-top:22px">
        <span id="regTxt">Tạo tài khoản</span>
      </button>
    </form>

    <div class="hint">
      Tài khoản mặc định: <code>admin</code> / <code>Admin@123</code><br>
      <span style="color:var(--text-soft)">(có thể đổi qua biến môi trường BC_ADMIN_USER / BC_ADMIN_PASS)</span>
    </div>

    <div class="secure">
      <span>🔐 PBKDF2-SHA256</span>
      <span>🛡️ SQL Injection-safe</span>
      <span>📋 Audit log</span>
    </div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);
const msg = (text, kind="err") => {
  const m = $("msg");
  m.className = "msg show msg--" + (kind === "ok" ? "ok" : "err");
  m.textContent = (kind === "ok" ? "✅ " : "⚠️ ") + text;
  clearTimeout(m._t);
  m._t = setTimeout(() => m.classList.remove("show"), 5000);
};

function switchTab(which){
  const isLogin = which === "login";
  $("tabLogin").classList.toggle("on", isLogin);
  $("tabRegister").classList.toggle("on", !isLogin);
  $("formLogin").hidden = !isLogin;
  $("formRegister").hidden = isLogin;
  $("msg").classList.remove("show");
}
$("tabLogin").addEventListener("click", () => switchTab("login"));
$("tabRegister").addEventListener("click", () => switchTab("register"));

function togglePw(inputId, btn){
  const el = $(inputId);
  el.type = el.type === "password" ? "text" : "password";
  btn.textContent = el.type === "password" ? "👁️" : "🙈";
}
$("togglePw1").addEventListener("click", (e) => togglePw("loginPass", e.currentTarget));
$("togglePw2").addEventListener("click", (e) => togglePw("regPass", e.currentTarget));

async function call(url, body, btn, txtEl, txtBusy){
  btn.disabled = true;
  const old = txtEl.textContent;
  txtEl.innerHTML = '<span class="spinner"></span>' + txtBusy;
  try {
    const res = await fetch(url, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify(body),
      credentials: "same-origin",
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
    return data;
  } finally {
    btn.disabled = false;
    txtEl.textContent = old;
  }
}

$("formLogin").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = { username: $("loginUser").value.trim(), password: $("loginPass").value };
  if (!body.username || body.password.length < 6) {
    return msg("Nhập tên đăng nhập và mật khẩu (≥6 ký tự).");
  }
  try {
    await call("/api/auth/login", body, $("btnLogin"), $("loginTxt"), "Đang xác thực…");
    msg("Đăng nhập thành công. Đang chuyển hướng…", "ok");
    setTimeout(() => location.href = "/", 450);
  } catch (err) {
    msg(err.message);
  }
});

$("formRegister").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    username: $("regUser").value.trim(),
    password: $("regPass").value,
    full_name: $("regName").value.trim(),
  };
  if (!body.username) return msg("Vui lòng nhập tên đăng nhập.");
  if (body.password.length < 6) return msg("Mật khẩu cần tối thiểu 6 ký tự.");
  try {
    await call("/api/auth/register", body, $("btnRegister"), $("regTxt"), "Đang tạo…");
    msg("Tạo tài khoản thành công. Đang chuyển hướng…", "ok");
    setTimeout(() => location.href = "/", 500);
  } catch (err) {
    msg(err.message);
  }
});
</script>
</body>
</html>
"""


# ==============================================================
# 4b. DASHBOARD TEMPLATE (đã beautify)
# ==============================================================
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
   Palette — 4 tông chính
   🔵 blue   → thông tin chung
   🟢 green  → Alive / Positive
   🔴 red    → Dead / Negative
   ⚪ gray   → nền / phụ trợ
   ============================================================ */
:root{
  --blue:#2563eb;--blue-dark:#1d4ed8;--blue-soft:#dbeafe;--blue-glow:rgba(37,99,235,.32);
  --green:#16a34a;--green-dark:#15803d;--green-soft:#dcfce7;
  --red:#dc2626;--red-dark:#b91c1c;--red-soft:#fee2e2;
  --gray:#64748b;--gray-dark:#475569;--gray-soft:#f1f5f9;

  --bg:#f8fafc;--surface:#fff;--surface-2:#f8fafc;--surface-3:#f1f5f9;
  --text:#0f172a;--text-muted:#64748b;--text-soft:#94a3b8;
  --border:#e2e8f0;--border-strong:#cbd5e1;

  --radius:12px;--radius-lg:18px;
  --shadow-sm:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04);
  --shadow:0 10px 30px -12px rgba(15,23,42,.15),0 2px 6px rgba(15,23,42,.05);
  --shadow-lg:0 24px 60px -20px rgba(15,23,42,.24);
  --font:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --transition:200ms cubic-bezier(.4,0,.2,1);
}
[data-theme="dark"]{
  --bg:#020617;--surface:#0f172a;--surface-2:#111c33;--surface-3:#1e293b;
  --text:#f1f5f9;--text-muted:#94a3b8;--text-soft:#64748b;
  --border:#1e293b;--border-strong:#334155;
  --blue:#60a5fa;--blue-soft:rgba(96,165,250,.14);--blue-glow:rgba(96,165,250,.35);
  --green:#4ade80;--green-soft:rgba(74,222,128,.14);
  --red:#f87171;--red-soft:rgba(248,113,113,.14);
  --gray:#94a3b8;--gray-soft:rgba(148,163,184,.14);
  --shadow:0 10px 30px -12px rgba(0,0,0,.5);
  --shadow-lg:0 24px 60px -20px rgba(0,0,0,.7);
}
*,*::before,*::after{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font-family:var(--font);font-size:14px;line-height:1.55;color:var(--text);
  background:var(--bg);padding:22px 16px 60px;-webkit-font-smoothing:antialiased;
  position:relative;overflow-x:hidden}
/* subtle ambient */
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(1000px 500px at 10% -10%,rgba(37,99,235,.08),transparent 60%),
    radial-gradient(900px 500px at 110% 10%,rgba(22,163,74,.06),transparent 60%);
}
h1,h2,h3,p{margin:0}
button,input,select{font:inherit;color:inherit}
button{cursor:pointer}
:focus-visible{outline:2px solid var(--blue);outline-offset:2px;border-radius:4px}

.app{max-width:1320px;margin:0 auto;display:flex;flex-direction:column;gap:16px;position:relative;z-index:1}

/* ---------- Header ---------- */
.header{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:24px 28px;box-shadow:var(--shadow);
  display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;
  position:relative;overflow:hidden}
.header::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(120deg,rgba(37,99,235,.06),transparent 40%,rgba(22,163,74,.05) 100%)}
.header__title{font-size:26px;font-weight:800;letter-spacing:-.6px;
  background:linear-gradient(120deg,var(--text) 30%,var(--text-muted) 130%);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.header__desc{font-size:13.5px;color:var(--text-muted);margin-top:6px;max-width:680px}
.header__actions{display:flex;align-items:center;gap:10px;position:relative;z-index:1}
.status{font-size:12px;font-weight:600;padding:7px 13px;border-radius:999px;
  background:var(--surface-3);color:var(--text-muted);border:1px solid var(--border);
  display:inline-flex;align-items:center;gap:7px;white-space:nowrap}
.status::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;
  box-shadow:0 0 0 0 currentColor;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 currentColor;opacity:1}50%{box-shadow:0 0 0 6px transparent;opacity:.6}}
.status.is-online{background:var(--green-soft);color:var(--green);border-color:var(--green)}
.status.is-offline{background:var(--red-soft);color:var(--red);border-color:var(--red)}
.status.is-checking{background:var(--gray-soft);color:var(--gray)}
.theme-btn{width:38px;height:38px;display:grid;place-items:center;background:var(--surface-3);
  border:1px solid var(--border);border-radius:10px;font-size:16px;transition:all var(--transition)}
.theme-btn:hover{border-color:var(--blue);color:var(--blue);transform:translateY(-1px)}
.user-chip{display:inline-flex;align-items:center;gap:8px;padding:6px 12px 6px 6px;
  background:var(--surface-3);border:1px solid var(--border);border-radius:999px;
  font-size:12px;font-weight:600;color:var(--text-muted)}
.user-chip__av{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;
  background:linear-gradient(135deg,var(--blue),var(--blue-dark));color:#fff;font-size:11px;font-weight:800}
.logout-btn{padding:7px 13px;font-size:12px;font-weight:600;background:transparent;
  border:1px solid var(--border);border-radius:10px;color:var(--text-muted);transition:all var(--transition)}
.logout-btn:hover{border-color:var(--red);color:var(--red);background:var(--red-soft)}

/* ---------- Tabs ---------- */
.tabs{display:flex;gap:4px;flex-wrap:wrap;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:6px;box-shadow:var(--shadow-sm);position:relative;overflow-x:auto}
.tab{display:inline-flex;align-items:center;gap:7px;padding:10px 15px;font-size:12.5px;
  font-weight:600;border-radius:var(--radius);border:none;background:transparent;
  color:var(--text-muted);transition:all var(--transition);white-space:nowrap;position:relative}
.tab:hover{background:var(--surface-3);color:var(--text)}
.tab.is-active{background:linear-gradient(135deg,var(--blue),var(--blue-dark));color:#fff;
  box-shadow:0 10px 22px -10px var(--blue-glow)}

/* ---------- Panels ---------- */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);
  box-shadow:var(--shadow);overflow:hidden;animation:panelIn .35s ease}
@keyframes panelIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.panel__head{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:15px 22px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.panel__head h2{font-size:15px;font-weight:700;display:flex;align-items:center;gap:9px}
.panel__head h2::before{content:"";width:4px;height:16px;border-radius:2px;
  background:linear-gradient(180deg,var(--blue),var(--blue-dark))}
.panel__body{padding:20px 22px}
.tabpanel{display:none;flex-direction:column;gap:16px}
.tabpanel.is-active{display:flex}

/* ---------- Stat cards ---------- */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
.stat{padding:18px 20px;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--surface);position:relative;overflow:hidden;transition:all var(--transition)}
.stat::before{content:"";position:absolute;inset:0;opacity:0;transition:opacity var(--transition);
  background:linear-gradient(135deg,rgba(37,99,235,.06),transparent)}
.stat:hover{box-shadow:var(--shadow);transform:translateY(-2px);border-color:var(--border-strong)}
.stat:hover::before{opacity:1}
.stat__icon{width:38px;height:38px;border-radius:11px;display:grid;place-items:center;
  font-size:18px;background:var(--surface-3);margin-bottom:10px;position:relative;z-index:1}
.stat__label{font-size:11px;font-weight:700;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:.08em;position:relative;z-index:1}
.stat__value{font-size:28px;font-weight:800;font-family:var(--mono);margin-top:4px;
  letter-spacing:-.6px;color:var(--text);position:relative;z-index:1}
.stat__value.blue{color:var(--blue)}
.stat__value.green{color:var(--green)}
.stat__value.red{color:var(--red)}
.stat__value.gray{color:var(--gray-dark)}

/* ---------- Chart cards ---------- */
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.chart-grid.two{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.chart-card{border:1px solid var(--border);border-radius:var(--radius);padding:18px;
  background:var(--surface);display:flex;flex-direction:column;gap:6px;transition:all var(--transition)}
.chart-card:hover{box-shadow:var(--shadow-sm)}
.chart-card__title{font-size:13.5px;font-weight:700;color:var(--text)}
.chart-card__desc{font-size:11.5px;color:var(--text-muted);line-height:1.55;margin-bottom:6px}
.chart-wrap{position:relative;height:270px}
.chart-wrap.tall{height:380px}

/* ---------- Table ---------- */
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--surface)}
table.dt{width:100%;border-collapse:collapse;font-size:12.5px;min-width:800px}
table.dt th,table.dt td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
table.dt th{background:var(--surface-3);font-weight:700;font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;color:var(--text-muted);position:sticky;top:0;cursor:pointer;user-select:none}
table.dt th:hover{color:var(--blue)}
table.dt tbody tr{transition:background var(--transition)}
table.dt tbody tr:hover{background:var(--surface-2)}
table.dt td.mono{font-family:var(--mono)}

/* ---------- Filter ---------- */
.filter-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.filter-field label{display:block;font-size:11px;font-weight:700;color:var(--text-muted);
  margin-bottom:6px;letter-spacing:.04em;text-transform:uppercase}
.filter-field input,.filter-field select{width:100%;padding:9px 11px;font-size:12.5px;
  background:var(--surface);border:1px solid var(--border);border-radius:9px;color:var(--text);
  transition:all var(--transition)}
.filter-field input:focus,.filter-field select:focus{outline:none;border-color:var(--blue);
  box-shadow:0 0 0 3px var(--blue-soft)}
.range-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.chip-group{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.chip{padding:6px 11px;font-size:11.5px;font-weight:600;border-radius:999px;
  border:1px solid var(--border);background:var(--surface);color:var(--text-muted);
  transition:all var(--transition)}
.chip:hover{border-color:var(--blue);color:var(--blue)}
.chip.is-on{background:var(--blue-soft);color:var(--blue);border-color:var(--blue);
  box-shadow:0 0 0 3px var(--blue-glow)}

/* ---------- Buttons ---------- */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:10px 18px;
  font-size:13px;font-weight:600;border-radius:10px;border:1px solid transparent;
  transition:all var(--transition)}
.btn:active:not(:disabled){transform:translateY(1px)}
.btn:disabled{opacity:.55;cursor:not-allowed}
.btn--primary{background:linear-gradient(135deg,var(--blue),var(--blue-dark));color:#fff;
  box-shadow:0 10px 22px -10px var(--blue-glow)}
.btn--primary:hover:not(:disabled){filter:brightness(1.05);transform:translateY(-1px)}
.btn--ghost{background:var(--surface);color:var(--text-muted);border-color:var(--border)}
.btn--ghost:hover:not(:disabled){border-color:var(--blue);color:var(--blue)}
.btn--sm{padding:7px 14px;font-size:12px}

/* ---------- Pagination ---------- */
.pagination{display:flex;align-items:center;justify-content:space-between;gap:10px;
  margin-top:12px;flex-wrap:wrap}
.pagination__info{font-size:12px;color:var(--text-muted)}
.pagination__btns{display:flex;gap:6px}
.pg-btn{padding:7px 13px;font-size:12px;font-weight:600;border-radius:9px;
  border:1px solid var(--border);background:var(--surface);color:var(--text-muted);
  transition:all var(--transition)}
.pg-btn:hover:not(:disabled){border-color:var(--blue);color:var(--blue)}
.pg-btn:disabled{opacity:.4;cursor:not-allowed}

/* ---------- Wizard ---------- */
.wizard{display:flex;flex-direction:column;gap:18px}
.steps{display:flex;justify-content:center;gap:12px;flex-wrap:wrap}
.step{display:flex;align-items:center;gap:10px;padding:11px 17px;border-radius:999px;
  border:1px solid var(--border);background:var(--surface-2);font-size:12.5px;
  font-weight:600;color:var(--text-muted);transition:all var(--transition)}
.step__num{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;
  background:var(--surface-3);font-size:11px;font-weight:700;color:var(--text-muted)}
.step.is-active{border-color:var(--blue);color:var(--blue);background:var(--blue-soft);
  box-shadow:0 0 0 4px var(--blue-glow)}
.step.is-active .step__num{background:var(--blue);color:#fff}
.step.is-done{border-color:var(--green);color:var(--green);background:var(--green-soft)}
.step.is-done .step__num{background:var(--green);color:#fff}
.wizard__body{padding:8px 4px}
.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.form-grid .field label{display:block;font-size:11.5px;font-weight:700;color:var(--text-muted);
  margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em}
.form-grid .field input,.form-grid .field select{width:100%;padding:11px 13px;font-size:13.5px;
  background:var(--surface);border:1px solid var(--border);border-radius:10px;color:var(--text);
  transition:all var(--transition)}
.form-grid .field input:focus,.form-grid .field select:focus{outline:none;border-color:var(--blue);
  box-shadow:0 0 0 3px var(--blue-soft)}
.wizard__nav{display:flex;justify-content:space-between;gap:10px;padding-top:14px;
  border-top:1px solid var(--border);margin-top:16px}

.predict-result{border-radius:var(--radius-lg);padding:26px;text-align:center;animation:fadeUp .4s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.predict-result--alive{background:var(--green-soft);border:1px solid var(--green);
  box-shadow:0 20px 50px -20px rgba(22,163,74,.5)}
.predict-result--dead{background:var(--red-soft);border:1px solid var(--red);
  box-shadow:0 20px 50px -20px rgba(220,38,38,.5)}
.predict-result__icon{font-size:56px;line-height:1;margin-bottom:10px}
.predict-result__label{font-size:28px;font-weight:800;letter-spacing:.06em;text-transform:uppercase}
.predict-result--alive .predict-result__label{color:var(--green-dark)}
.predict-result--dead .predict-result__label{color:var(--red-dark)}
.predict-result__prob{font-size:14px;margin-top:8px;color:var(--text-muted)}
.predict-result__prob strong{font-size:22px;font-family:var(--mono);color:var(--text);margin-left:6px}
.predict-meta{margin-top:14px;padding:14px 18px;background:var(--surface-2);
  border-radius:12px;text-align:left;font-size:12px;color:var(--text-muted);border:1px solid var(--border)}
.predict-meta strong{color:var(--text)}

/* ---------- SVM form ---------- */
.groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;padding:20px 22px}
.group{border:1px solid var(--border);border-radius:var(--radius);padding:16px;
  background:var(--surface-2);transition:all var(--transition)}
.group:hover{border-color:var(--border-strong)}
.group__title{font-size:11.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--blue);margin-bottom:12px;display:flex;align-items:center;gap:8px}
.group__title::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--blue);
  box-shadow:0 0 0 4px var(--blue-glow)}
.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:10px}
.field label{display:block;font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:5px}
.field input{width:100%;padding:10px 12px;font-size:13px;font-family:var(--mono);
  background:var(--surface);border:1px solid var(--border);border-radius:9px;color:var(--text);
  transition:all var(--transition)}
.field input:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 3px var(--blue-soft)}
.field input.is-invalid{border-color:var(--red);background:var(--red-soft)}
.actions{display:flex;gap:10px;flex-wrap:wrap;padding:18px 22px;border-top:1px solid var(--border);
  background:var(--surface-2)}
.result__empty{padding:46px 22px;text-align:center;color:var(--text-muted)}
.result__empty .icon{font-size:46px;opacity:.3;display:block;margin-bottom:12px}
.result__card{padding:22px}
.verdict{border-radius:var(--radius-lg);padding:26px 16px;text-align:center;margin-bottom:18px}
.verdict--malignant{background:var(--red-soft);border:1px solid var(--red);
  box-shadow:0 20px 50px -20px rgba(220,38,38,.5)}
.verdict--benign{background:var(--green-soft);border:1px solid var(--green);
  box-shadow:0 20px 50px -20px rgba(22,163,74,.5)}
.verdict__icon{font-size:44px;line-height:1;margin-bottom:10px}
.verdict__label{font-size:22px;font-weight:800;letter-spacing:.05em;text-transform:uppercase}
.verdict--malignant .verdict__label{color:var(--red-dark)}
.verdict--benign .verdict__label{color:var(--green-dark)}
.verdict__sub{font-size:12.5px;color:var(--text-muted);margin-top:4px}
.prob{margin-bottom:16px}
.prob__head{display:flex;justify-content:space-between;font-size:12px;font-weight:600;margin-bottom:6px}
.prob__track{height:8px;border-radius:999px;background:var(--surface-3);overflow:hidden}
.prob__fill{height:100%;border-radius:999px;transition:width .65s cubic-bezier(.2,.9,.3,1.1)}
.prob__fill--red{background:linear-gradient(90deg,#f87171,var(--red))}
.prob__fill--green{background:linear-gradient(90deg,#4ade80,var(--green))}
.meta__row{display:flex;justify-content:space-between;gap:12px;padding:10px 0;font-size:13px;
  border-bottom:1px dashed var(--border)}
.meta__row:last-child{border-bottom:none}
.meta__key{color:var(--text-muted)}
.meta__val{font-weight:600;font-family:var(--mono)}
.alert{margin:0;padding:14px 16px;border-radius:var(--radius);
  background:var(--red-soft);border:1px solid var(--red);color:var(--red-dark);
  font-size:13px;line-height:1.6}
.note{margin-top:14px;padding:12px 15px;font-size:11.5px;line-height:1.65;color:var(--text-muted);
  background:var(--surface-2);border-left:3px solid var(--border-strong);
  border-radius:0 10px 10px 0}
.spinner{width:14px;height:14px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;
  border-radius:50%;animation:spin .7s linear infinite;display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
.footer{text-align:center;font-size:12px;color:var(--text-muted);padding-top:8px}
.muted{color:var(--text-muted);font-size:12.5px}

/* ---------- Toast ---------- */
.toast-stack{position:fixed;top:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px}
.toast{padding:12px 16px;border-radius:11px;background:var(--surface);
  border:1px solid var(--border);box-shadow:var(--shadow-lg);font-size:13px;
  display:flex;align-items:flex-start;gap:10px;max-width:340px;animation:slideIn .3s ease}
.toast--ok{border-color:var(--green);background:var(--green-soft);color:var(--green-dark)}
.toast--err{border-color:var(--red);background:var(--red-soft);color:var(--red-dark)}
.toast--info{border-color:var(--blue);background:var(--blue-soft);color:var(--blue-dark)}
@keyframes slideIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:none}}

/* ---------- Skeleton ---------- */
.skeleton{background:linear-gradient(90deg,var(--surface-3) 25%,var(--surface-2) 50%,var(--surface-3) 75%);
  background-size:200% 100%;animation:shimmer 1.4s ease-in-out infinite;border-radius:8px}
@keyframes shimmer{from{background-position:200% 0}to{background-position:-200% 0}}

/* ---------- Responsive ---------- */
@media (max-width:640px){
  .header{padding:18px 18px}
  .header__title{font-size:20px}
  .stat__value{font-size:24px}
  .panel__body{padding:16px}
}
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
      <span class="user-chip" id="userChip" hidden>
        <span class="user-chip__av" id="userAv">U</span>
        <span id="userName">user</span>
      </span>
      <button class="theme-btn" id="btnTheme" type="button" title="Đổi sáng/tối">🌙</button>
      <button class="logout-btn" id="btnLogout" type="button">Đăng xuất</button>
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
       TAB 8 — SVM 30 đặc trưng
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

<div class="toast-stack" id="toastStack"></div>

<script>
/* ============================================================
   PALETTE
   ============================================================ */
const CLR = {
  blue:  "#2563eb", green: "#16a34a", red: "#dc2626", gray: "#94a3b8",
  blueSoft:  "rgba(37,99,235,.65)",
  greenSoft: "rgba(22,163,74,.75)",
  redSoft:   "rgba(220,38,38,.75)",
  graySoft:  "rgba(148,163,184,.7)",
};

/* ============================================================
   CONFIG
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
   TOAST
   ============================================================ */
const Toast = (() => {
  const stack = () => document.getElementById("toastStack");
  function show(text, kind="info", ttl=4200){
    const el = document.createElement("div");
    el.className = "toast toast--" + kind;
    const icon = kind === "ok" ? "✅" : kind === "err" ? "⚠️" : "ℹ️";
    el.innerHTML = `<span style="font-size:16px">${icon}</span><span>${text}</span>`;
    stack().appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity .3s, transform .3s";
      el.style.opacity = "0"; el.style.transform = "translateX(20px)";
      setTimeout(() => el.remove(), 300);
    }, ttl);
  }
  return { show };
})();

/* ============================================================
   API
   ============================================================ */
const Api = (() => {
  async function request(url, options={}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch(url, {...options, signal:controller.signal, credentials:"same-origin"});
      if (res.status === 401) {
        Toast.show("Phiên đăng nhập đã hết hạn. Đang chuyển về trang đăng nhập…", "err");
        setTimeout(() => location.href = "/login", 900);
        throw new Error("Unauthorized");
      }
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
    me:       () => request("/api/auth/me"),
    logout:   () => post("/api/auth/logout", {}),
  };
})();

/* ============================================================
   CHART REGISTRY
   ============================================================ */
const Charts = (() => {
  const registry = {};
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
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
  function updateAll() {
    Object.values(registry).forEach(c => {
      try {
        if (c.options?.plugins?.legend?.labels) c.options.plugins.legend.labels.color = textColor();
        if (c.options?.scales) {
          Object.values(c.options.scales).forEach(s => {
            if (s.ticks) s.ticks.color = textColor();
            if (s.grid) s.grid.color = gridColor();
          });
        }
        c.update("none");
      } catch {}
    });
  }
  return {make, destroy, baseOpts, gridColor, textColor, updateAll};
})();

/* ============================================================
   UI
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
    setTimeout(() => Charts.updateAll(), 60);
  }

  async function loadUser() {
    try {
      const d = await Api.me();
      const u = d.user;
      $("userChip").hidden = false;
      $("userName").textContent = u.full_name || u.username;
      $("userAv").textContent = (u.username || "U").slice(0,1).toUpperCase();
    } catch {}
  }

  return { buildForm, readForm, fillSample, clearResult, showError, showResult, setLoading, setStatus, applyTheme, loadUser };
})();

/* ============================================================
   TABS
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
   DASHBOARD
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
    if (!rows || !rows.length) { el.innerHTML = `<div style="padding:22px" class="muted">Không có dữ liệu.</div>`; return; }
    const head = columns.map(c => `<th>${c}</th>`).join("");
    const body = rows.map(r => `<tr>${columns.map(c => {
      const v = r[c]; const isNum = typeof v === "number";
      return `<td class="${isNum?'mono':''}">${v === null || v === undefined ? "" : v}</td>`;
    }).join("")}</tr>`).join("");
    el.innerHTML = `<table class="dt"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  async function loadOverview() {
    try {
      const d = await Api.overview();
      $("ovStats").innerHTML =
        statCard("👥", "Tổng bệnh nhân", d.rows, "blue") +
        statCard("🟢", "Alive", d.alive, "green") +
        statCard("🔴", "Dead", d.dead, "red") +
        statCard("📈", "Thời gian sống TB", num(d.avg_survival, 1) + " th", "gray");

      Charts.make("ovStatus", {
        type:"doughnut",
        data:{
          labels:["Alive","Dead"],
          datasets:[{data:[d.alive, d.dead],
            backgroundColor:[CLR.green, CLR.red], borderWidth:0,
            hoverOffset:6}]
        },
        options:{responsive:true, maintainAspectRatio:false, cutout:"64%",
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

      const stageKeys = Object.keys(d.stage_distribution || {}).sort();
      Charts.make("ovStage", {
        type:"bar",
        data:{labels: stageKeys,
          datasets:[{label:"Số bệnh nhân",
            data: stageKeys.map(k => d.stage_distribution[k]),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });
    } catch(e) { $("ovStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  async function loadMeta() {
    if (state.meta) return state.meta;
    state.meta = await Api.meta();
    return state.meta;
  }

  function buildFilterUI() {
    const m = state.meta; if (!m) return;
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
          backgroundColor:estColors, borderWidth:0, hoverOffset:6}]},
        options:{responsive:true, maintainAspectRatio:false, cutout:"64%",
          plugins:{legend:{position:"bottom", labels:{color:Charts.textColor()}}}}
      });

      const progLabels = Object.keys(d.progesterone_dist);
      const progColors = progLabels.map(l => l.toLowerCase() === "positive" ? CLR.green : CLR.red);
      Charts.make("bioProg", {
        type:"doughnut",
        data:{labels:progLabels, datasets:[{data:Object.values(d.progesterone_dist),
          backgroundColor:progColors, borderWidth:0, hoverOffset:6}]},
        options:{responsive:true, maintainAspectRatio:false, cutout:"64%",
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

  async function loadSurvival() {
    try {
      const d = await Api.survival({filters: state.filters});
      $("svStats").innerHTML =
        statCard("⏱️","Tổng", d.total, "blue") +
        statCard("📈","Tháng TB", num(d.survival_stats.mean, 1), "gray") +
        statCard("📊","Trung vị", num(d.survival_stats.median, 1), "gray") +
        statCard("⬇️","Min", num(d.survival_stats.min, 0), "gray") +
        statCard("⬆️","Max", num(d.survival_stats.max, 0), "gray");

      Charts.make("svHist", {
        type:"bar",
        data:{labels:Object.keys(d.survival_hist),
          datasets:[{label:"Số ca", data:Object.values(d.survival_hist),
            backgroundColor:CLR.blue, borderRadius:6}]},
        options:Charts.baseOpts()
      });

      const stages = Object.keys(d.survival_boxplot_by_stage || {});
      const boxes = stages.map(s => {
        const b = d.survival_boxplot_by_stage[s];
        return {min:b.min, q1:b.q1, median:b.median, q3:b.q3, max:b.max};
      });

      if (window.Chart && stages.length) {
        Charts.make("svBox", {
          type:"boxplot",
          data:{labels: stages,
            datasets:[{label:"Thời gian sống (tháng)", data: boxes,
              backgroundColor: CLR.blueSoft, borderColor: CLR.blue,
              borderWidth: 1.5, medianColor: CLR.blue,
              itemRadius: 0, outlierRadius: 2, outlierBackgroundColor: CLR.red}]},
          options:{responsive:true, maintainAspectRatio:false,
            plugins:{legend:{display:false}},
            scales:{
              x:{ticks:{color:Charts.textColor()}, grid:{color:Charts.gridColor()}},
              y:{ticks:{color:Charts.textColor()}, grid:{color:Charts.gridColor()}, beginAtZero:true}
            }}
        });
      }

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
        options:{...Charts.baseOpts(), plugins:{legend:{labels:{color:Charts.textColor()}}}}
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
      }
    } catch(e) { $("svStats").innerHTML = `<div class="alert">${e.message}</div>`; }
  }

  async function ensureModel() {
    try { return await Api.train(); } catch(e) { return null; }
  }

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
   WIZARD
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
   APP
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
      Toast.show("Phân tích hoàn tất.", "ok");
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
    document.getElementById("btnLogout").addEventListener("click", async () => {
      try { await Api.logout(); } catch {}
      Toast.show("Đã đăng xuất.", "ok");
      setTimeout(() => location.href = "/login", 400);
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
    UI.loadUser();
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
