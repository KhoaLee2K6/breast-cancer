from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import numpy as np
import joblib
import uvicorn


# ==============================================================
# 1. MODEL  —  graceful loading
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
# 2. SCHEMAS  —  30 đặc trưng load_breast_cancer
# ==============================================================
class BreastCancerInput(BaseModel):
    # --- mean ---
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
    # --- error ---
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
    # --- worst ---
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


CANCER_CLASSES = {0: "malignant", 1: "benign"}
CLASS_LABELS   = ["malignant", "benign"]


# ==============================================================
# 3. FASTAPI APP
# ==============================================================
app = FastAPI(
    title="Breast Cancer SVM Diagnostic API",
    description="SVM classifier cho bộ dữ liệu Breast Cancer (30 đặc trưng).",
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
    }


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


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_TEMPLATE


# ==============================================================
# 4. FRONTEND  —  single-page, modular, design tokens
# ==============================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi" data-theme="light">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Breast Cancer · SVM Diagnostic</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
/* ============================================================
   1. DESIGN TOKENS
   ============================================================ */
:root {
  --bg-1: #eef2f8;
  --bg-2: #e0e7ff;
  --bg-3: #f0f9ff;

  --surface: #ffffff;
  --surface-2: #f8fafc;
  --surface-3: #f1f5f9;

  --text: #0f172a;
  --text-muted: #64748b;
  --text-soft: #94a3b8;

  --border: #e2e8f0;
  --border-strong: #cbd5e1;

  --primary: #2563eb;
  --primary-dark: #1d4ed8;
  --primary-soft: #dbeafe;
  --primary-glow: rgba(37, 99, 235, .25);

  --violet: #7c3aed;
  --violet-soft: #ede9fe;

  --danger: #dc2626;
  --danger-dark: #b91c1c;
  --danger-soft: #fef2f2;
  --danger-border: #fecaca;

  --success: #16a34a;
  --success-dark: #15803d;
  --success-soft: #f0fdf4;
  --success-border: #bbf7d0;

  --warn: #d97706;
  --warn-soft: #fffbeb;

  --radius-sm: 8px;
  --radius: 12px;
  --radius-lg: 16px;
  --radius-xl: 22px;

  --shadow-xs: 0 1px 2px rgba(15, 23, 42, .05);
  --shadow-sm: 0 1px 3px rgba(15, 23, 42, .06), 0 1px 2px rgba(15, 23, 42, .04);
  --shadow:    0 4px 12px rgba(15, 23, 42, .08), 0 2px 4px rgba(15, 23, 42, .04);
  --shadow-lg: 0 20px 40px -16px rgba(15, 23, 42, .18), 0 8px 16px -8px rgba(15, 23, 42, .08);

  --font: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;

  --transition: 180ms cubic-bezier(0.4, 0, 0.2, 1);
}

[data-theme="dark"] {
  --bg-1: #020617;
  --bg-2: #0b132b;
  --bg-3: #1e1b4b;

  --surface: #0f172a;
  --surface-2: #111c33;
  --surface-3: #1e293b;

  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --text-soft: #64748b;

  --border: #1e293b;
  --border-strong: #334155;

  --primary: #60a5fa;
  --primary-dark: #3b82f6;
  --primary-soft: rgba(96, 165, 250, .14);
  --primary-glow: rgba(96, 165, 250, .3);

  --violet: #a78bfa;
  --violet-soft: rgba(167, 139, 250, .14);

  --danger: #f87171;
  --danger-dark: #ef4444;
  --danger-soft: rgba(248, 113, 113, .1);
  --danger-border: rgba(248, 113, 113, .3);

  --success: #4ade80;
  --success-dark: #22c55e;
  --success-soft: rgba(74, 222, 128, .1);
  --success-border: rgba(74, 222, 128, .3);
}

/* ============================================================
   2. RESET / BASE
   ============================================================ */
*, *::before, *::after { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.55;
  color: var(--text);
  background:
    radial-gradient(1200px 620px at 6% -10%, var(--bg-2) 0%, transparent 60%),
    radial-gradient(1000px 520px at 100% 0%, var(--bg-3) 0%, transparent 55%),
    var(--bg-1);
  background-attachment: fixed;
  padding: 22px 16px 60px;
  -webkit-font-smoothing: antialiased;
  transition: background var(--transition), color var(--transition);
}
h1, h2, h3, p { margin: 0; }
button, input, select { font: inherit; color: inherit; }
button { cursor: pointer; }
:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; border-radius: 4px; }

/* ============================================================
   3. LAYOUT
   ============================================================ */
.app {
  max-width: 1240px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 20px;
  align-items: start;
}
@media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }

/* ============================================================
   4. HEADER
   ============================================================ */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px 22px;
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
}
.header::before {
  content: "";
  position: absolute; inset: 0;
  background: linear-gradient(120deg, transparent 40%, var(--primary-soft) 100%);
  opacity: .55;
  pointer-events: none;
}
.brand { display: flex; align-items: center; gap: 14px; position: relative; z-index: 1; }
.brand__icon {
  width: 48px; height: 48px; flex: 0 0 48px;
  display: grid; place-items: center;
  border-radius: 14px;
  background: linear-gradient(135deg, var(--primary), var(--violet));
  color: #fff;
  font-size: 22px;
  box-shadow: 0 10px 20px -6px var(--primary-glow);
}
.brand__title { font-size: 19px; font-weight: 800; letter-spacing: -.3px; }
.brand__sub   { font-size: 12.5px; color: var(--text-muted); margin-top: 2px; }
.header__actions { display: flex; align-items: center; gap: 10px; position: relative; z-index: 1; }

.status {
  font-size: 12px; font-weight: 600;
  padding: 7px 13px;
  border-radius: 999px;
  background: var(--surface-3);
  color: var(--text-muted);
  border: 1px solid var(--border);
  white-space: nowrap;
  transition: all var(--transition);
  display: inline-flex; align-items: center; gap: 6px;
}
.status::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 6px currentColor;
}
.status.is-online  { background: var(--success-soft); color: var(--success); border-color: var(--success-border); }
.status.is-offline { background: var(--danger-soft);  color: var(--danger);  border-color: var(--danger-border); }
.status.is-checking{ background: var(--warn-soft);    color: var(--warn);    border-color: #fde68a; }

.theme-btn {
  width: 38px; height: 38px;
  display: grid; place-items: center;
  background: var(--surface-3);
  border: 1px solid var(--border);
  border-radius: 10px;
  font-size: 16px;
  transition: all var(--transition);
}
.theme-btn:hover { background: var(--primary-soft); border-color: var(--primary); transform: translateY(-1px); }

/* ============================================================
   5. PANEL / CARD
   ============================================================ */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.panel__head {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, var(--surface), var(--surface-2));
}
.panel__head h2 { font-size: 14.5px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
.panel__head h2::before {
  content: ""; width: 4px; height: 16px; border-radius: 2px;
  background: linear-gradient(180deg, var(--primary), var(--violet));
}
.panel__body { padding: 18px 20px; }

/* ============================================================
   6. FORM GROUPS
   ============================================================ */
.groups {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 14px;
  padding: 18px 20px;
}
.group {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px;
  background: var(--surface-2);
  transition: all var(--transition);
}
.group:hover { border-color: var(--border-strong); }
.group__title {
  font-size: 11.5px; font-weight: 700;
  letter-spacing: .09em; text-transform: uppercase;
  color: var(--primary);
  margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}
.group__title::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-soft);
}

.fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));
  gap: 10px;
}
.field label {
  display: block;
  font-size: 11px; font-weight: 600;
  color: var(--text-muted);
  margin-bottom: 5px;
  letter-spacing: .02em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.field input {
  width: 100%;
  padding: 9px 11px;
  font-size: 13px;
  font-family: var(--mono);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  transition: all var(--transition);
}
.field input:hover { border-color: var(--border-strong); }
.field input:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-glow);
}
.field input.is-invalid {
  border-color: var(--danger);
  background: var(--danger-soft);
}
.field input::-webkit-outer-spin-button,
.field input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.field input[type=number] { -moz-appearance: textfield; }

/* ============================================================
   7. ACTIONS
   ============================================================ */
.actions {
  display: flex; gap: 10px; flex-wrap: wrap;
  padding: 16px 20px;
  border-top: 1px solid var(--border);
  background: var(--surface-2);
}
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  padding: 10px 20px;
  font-size: 13.5px; font-weight: 600;
  border-radius: var(--radius);
  border: 1px solid transparent;
  transition: all var(--transition);
  user-select: none;
  white-space: nowrap;
}
.btn:active:not(:disabled) { transform: translateY(1px); }
.btn:disabled { opacity: .55; cursor: not-allowed; }

.btn--primary {
  background: linear-gradient(135deg, var(--primary), var(--violet));
  color: #fff;
  box-shadow: 0 8px 20px -8px var(--primary-glow);
  padding-inline: 28px;
}
.btn--primary:hover:not(:disabled) {
  box-shadow: 0 12px 26px -8px var(--primary-glow);
  transform: translateY(-1px);
}
.btn--ghost {
  background: var(--surface);
  color: var(--text-muted);
  border-color: var(--border);
}
.btn--ghost:hover:not(:disabled) {
  background: var(--surface-3);
  color: var(--text);
  border-color: var(--border-strong);
}
.btn--danger-ghost {
  background: var(--danger-soft);
  color: var(--danger);
  border-color: var(--danger-border);
}
.btn--danger-ghost:hover:not(:disabled) { background: var(--danger); color: #fff; }
.btn--success-ghost {
  background: var(--success-soft);
  color: var(--success);
  border-color: var(--success-border);
}
.btn--success-ghost:hover:not(:disabled) { background: var(--success); color: #fff; }
.btn--sm { padding: 7px 14px; font-size: 12px; }

.spinner {
  width: 14px; height: 14px;
  border: 2px solid rgba(255,255,255,.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ============================================================
   8. RESULT PANEL
   ============================================================ */
.result { position: sticky; top: 20px; }

.result__empty {
  padding: 46px 20px;
  text-align: center;
  color: var(--text-muted);
}
.result__empty .icon {
  font-size: 40px; opacity: .3;
  display: block; margin-bottom: 12px;
}
.result__empty p { font-size: 13px; line-height: 1.7; }

.result__card { padding: 20px; animation: fadeUp .35s cubic-bezier(.2,.9,.3,1); }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: none; }
}

.verdict {
  border-radius: var(--radius-lg);
  padding: 24px 16px;
  text-align: center;
  margin-bottom: 18px;
  position: relative;
  overflow: hidden;
}
.verdict::before {
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 0%, rgba(255,255,255,.35), transparent 65%);
  pointer-events: none;
}
.verdict--malignant { background: var(--danger-soft);  border: 1px solid var(--danger-border); }
.verdict--benign    { background: var(--success-soft); border: 1px solid var(--success-border); }
.verdict__icon  { font-size: 40px; line-height: 1; margin-bottom: 10px; position: relative; }
.verdict__label { font-size: 20px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; position: relative; }
.verdict--malignant .verdict__label { color: var(--danger); }
.verdict--benign    .verdict__label { color: var(--success); }
.verdict__sub   { font-size: 12.5px; color: var(--text-muted); margin-top: 4px; letter-spacing: .04em; position: relative; }

/* probability bar */
.prob { margin-bottom: 16px; }
.prob__head { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; margin-bottom: 6px; }
.prob__track {
  height: 8px; border-radius: 999px;
  background: var(--surface-3);
  overflow: hidden;
}
.prob__fill {
  height: 100%; border-radius: 999px;
  transition: width .55s cubic-bezier(.2,.9,.3,1);
}
.prob__fill--danger  { background: linear-gradient(90deg, #f87171, #dc2626); }
.prob__fill--success { background: linear-gradient(90deg, #4ade80, #16a34a); }

.meta { display: flex; flex-direction: column; gap: 0; }
.meta__row {
  display: flex; justify-content: space-between; gap: 12px;
  padding: 10px 0;
  font-size: 13px;
  border-bottom: 1px dashed var(--border);
}
.meta__row:last-child { border-bottom: none; }
.meta__key { color: var(--text-muted); }
.meta__val { font-weight: 600; font-family: var(--mono); }

.alert {
  margin: 0 20px 18px;
  padding: 14px 16px;
  border-radius: var(--radius);
  background: var(--danger-soft);
  border: 1px solid var(--danger-border);
  color: var(--danger-dark);
  font-size: 13px;
  line-height: 1.6;
  animation: fadeUp .3s;
}
[data-theme="dark"] .alert { color: #fecaca; }
.alert strong { display: block; margin-bottom: 4px; }

.note {
  margin-top: 14px;
  padding: 11px 14px;
  font-size: 11.5px; line-height: 1.65;
  color: var(--text-muted);
  background: var(--surface-2);
  border-left: 3px solid var(--border-strong);
  border-radius: 0 8px 8px 0;
}

/* ============================================================
   9. FOOTER
   ============================================================ */
.footer {
  text-align: center;
  font-size: 12px;
  color: var(--text-muted);
  padding-top: 6px;
}

/* ============================================================
   10. RESPONSIVE
   ============================================================ */
@media (max-width: 640px) {
  body { padding: 14px 10px 40px; }
  .header { padding: 14px 16px; }
  .brand__title { font-size: 16px; }
  .panel__body, .groups, .actions { padding: 14px 14px; }
  .btn { flex: 1 1 auto; }
}
</style>
</head>

<body>
<div class="app">

  <!-- ============== HEADER ============== -->
  <header class="header">
    <div class="brand">
      <div class="brand__icon" aria-hidden="true">⚕</div>
      <div>
        <h1 class="brand__title">Chẩn đoán ung thư vú</h1>
        <p class="brand__sub">Mô hình SVM · 30 đặc trưng tế bào (FNA)</p>
      </div>
    </div>
    <div class="header__actions">
      <span class="status" id="status">Đang kiểm tra…</span>
      <button class="theme-btn" id="btnTheme" type="button" title="Đổi chế độ sáng/tối">🌙</button>
    </div>
  </header>

  <!-- ============== MAIN ============== -->
  <div class="layout">

    <!-- ---------- FORM ---------- -->
    <form class="panel" id="form" autocomplete="off" novalidate>
      <div class="panel__head">
        <h2>Thông số xét nghiệm</h2>
        <div style="display:flex; gap:6px;">
          <button type="button" class="btn btn--danger-ghost btn--sm" id="btnSampleMalignant">⚠️ Mẫu ác tính</button>
          <button type="button" class="btn btn--success-ghost btn--sm" id="btnSampleBenign">✅ Mẫu lành tính</button>
        </div>
      </div>

      <div class="groups" id="groups"><!-- sinh bằng JS --></div>

      <div class="actions">
        <button type="submit" class="btn btn--primary" id="btnSubmit">
          <span class="spinner" id="spinner" hidden></span>
          <span id="btnText">🔬 Phân tích</span>
        </button>
        <button type="reset" class="btn btn--ghost" id="btnReset">🧹 Xoá tất cả</button>
      </div>
    </form>

    <!-- ---------- RESULT ---------- -->
    <aside class="panel result" id="result">
      <div class="panel__head"><h2>Kết quả</h2></div>

      <div class="result__empty" id="resultEmpty">
        <span class="icon" aria-hidden="true">🔬</span>
        <p>Nhập đầy đủ <strong>30 chỉ số</strong><br>rồi bấm <strong>Phân tích</strong>.</p>
      </div>

      <div id="resultBody" hidden></div>
      <div id="resultError" hidden></div>
    </aside>

  </div>

  <p class="footer">Demo học thuật · Không thay thế chẩn đoán y khoa</p>
</div>


<script>
/* =====================================================================
   MODULE 1 · CONFIG
   Nguồn duy nhất mô tả 30 đặc trưng — khớp 100% với
   class BreastCancerInput trong FastAPI.
   ===================================================================== */
const CONFIG = {
  API_URL:    "/predict",
  HEALTH_URL: "/health",
  REQUEST_TIMEOUT_MS: 15000,
};

const FEATURE_GROUPS = [
  {
    key: "mean",
    title: "Giá trị trung bình",
    fields: [
      ["mean_radius",            "Bán kính"],
      ["mean_texture",           "Kết cấu"],
      ["mean_perimeter",         "Chu vi"],
      ["mean_area",              "Diện tích"],
      ["mean_smoothness",        "Độ mịn"],
      ["mean_compactness",       "Độ đặc"],
      ["mean_concavity",         "Độ lõm"],
      ["mean_concave_points",    "Điểm lõm"],
      ["mean_symmetry",          "Đối xứng"],
      ["mean_fractal_dimension", "Fractal"],
    ],
  },
  {
    key: "error",
    title: "Sai số chuẩn",
    fields: [
      ["radius_error",            "Bán kính"],
      ["texture_error",           "Kết cấu"],
      ["perimeter_error",         "Chu vi"],
      ["area_error",              "Diện tích"],
      ["smoothness_error",        "Độ mịn"],
      ["compactness_error",       "Độ đặc"],
      ["concavity_error",         "Độ lõm"],
      ["concave_points_error",    "Điểm lõm"],
      ["symmetry_error",          "Đối xứng"],
      ["fractal_dimension_error", "Fractal"],
    ],
  },
  {
    key: "worst",
    title: "Giá trị xấu nhất",
    fields: [
      ["worst_radius",            "Bán kính"],
      ["worst_texture",           "Kết cấu"],
      ["worst_perimeter",         "Chu vi"],
      ["worst_area",              "Diện tích"],
      ["worst_smoothness",        "Độ mịn"],
      ["worst_compactness",       "Độ đặc"],
      ["worst_concavity",         "Độ lõm"],
      ["worst_concave_points",    "Điểm lõm"],
      ["worst_symmetry",          "Đối xứng"],
      ["worst_fractal_dimension", "Fractal"],
    ],
  },
];

/* Mẫu tham chiếu lấy từ load_breast_cancer() (sklearn) */
const SAMPLES = {
  malignant: {
    mean_radius: 17.99, mean_texture: 10.38, mean_perimeter: 122.8, mean_area: 1001.0,
    mean_smoothness: 0.1184, mean_compactness: 0.2776, mean_concavity: 0.3001,
    mean_concave_points: 0.1471, mean_symmetry: 0.2419, mean_fractal_dimension: 0.07871,
    radius_error: 1.095, texture_error: 0.9053, perimeter_error: 8.589, area_error: 153.4,
    smoothness_error: 0.006399, compactness_error: 0.04904, concavity_error: 0.05373,
    concave_points_error: 0.01587, symmetry_error: 0.03003, fractal_dimension_error: 0.006193,
    worst_radius: 25.38, worst_texture: 17.33, worst_perimeter: 184.6, worst_area: 2019.0,
    worst_smoothness: 0.1622, worst_compactness: 0.6656, worst_concavity: 0.7119,
    worst_concave_points: 0.2654, worst_symmetry: 0.4601, worst_fractal_dimension: 0.1189,
  },
  benign: {
    mean_radius: 13.54, mean_texture: 14.36, mean_perimeter: 87.46, mean_area: 566.3,
    mean_smoothness: 0.09779, mean_compactness: 0.08129, mean_concavity: 0.06664,
    mean_concave_points: 0.04781, mean_symmetry: 0.1885, mean_fractal_dimension: 0.05766,
    radius_error: 0.2699, texture_error: 0.7886, perimeter_error: 2.058, area_error: 23.56,
    smoothness_error: 0.008462, compactness_error: 0.0146, concavity_error: 0.02387,
    concave_points_error: 0.01315, symmetry_error: 0.0198, fractal_dimension_error: 0.0023,
    worst_radius: 15.11, worst_texture: 19.26, worst_perimeter: 99.7, worst_area: 711.2,
    worst_smoothness: 0.144, worst_compactness: 0.1773, worst_concavity: 0.239,
    worst_concave_points: 0.1288, worst_symmetry: 0.2977, worst_fractal_dimension: 0.07259,
  },
};

const LABELS = {
  0: { en: "Malignant", vi: "Ác tính",   icon: "⚠️", cls: "malignant" },
  1: { en: "Benign",    vi: "Lành tính", icon: "✅", cls: "benign"    },
};


/* =====================================================================
   MODULE 2 · API  —  tách hoàn toàn khỏi UI
   ===================================================================== */
const Api = (() => {
  async function request(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      const text = await res.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { data = text; }
      if (!res.ok) {
        const detail = (data && data.detail) ? data.detail : (typeof data === "string" ? data : res.statusText);
        throw new Error(`HTTP ${res.status} — ${detail}`);
      }
      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    predict(payload) {
      return request(CONFIG.API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    health() { return request(CONFIG.HEALTH_URL); },
  };
})();


/* =====================================================================
   MODULE 3 · UI  —  chỉ thao tác DOM
   ===================================================================== */
const UI = (() => {
  const $ = (id) => document.getElementById(id);

  /* ---------- Sinh form ---------- */
  function buildForm() {
    $("groups").innerHTML = FEATURE_GROUPS.map((g) => `
      <section class="group">
        <h3 class="group__title">${g.title}</h3>
        <div class="fields">
          ${g.fields.map(([name, label]) => `
            <div class="field">
              <label for="${name}" title="${name}">${label}</label>
              <input id="${name}" name="${name}" type="number"
                     step="any" min="0" placeholder="0.0" required />
            </div>
          `).join("")}
        </div>
      </section>
    `).join("");
  }

  /* ---------- Đọc form + validate ---------- */
  function readForm() {
    const payload = {};
    let firstInvalid = null;
    for (const g of FEATURE_GROUPS) {
      for (const [name] of g.fields) {
        const el = $(name);
        el.classList.remove("is-invalid");
        const raw = el.value.trim();
        if (raw === "") {
          el.classList.add("is-invalid");
          if (!firstInvalid) firstInvalid = { name, reason: "bỏ trống" };
          continue;
        }
        const val = Number(raw);
        if (!Number.isFinite(val)) {
          el.classList.add("is-invalid");
          if (!firstInvalid) firstInvalid = { name, reason: "không phải số" };
          continue;
        }
        if (val < 0) {
          el.classList.add("is-invalid");
          if (!firstInvalid) firstInvalid = { name, reason: "âm" };
          continue;
        }
        payload[name] = val;
      }
    }
    if (firstInvalid) {
      const field = firstInvalid.name;
      const label = (FEATURE_GROUPS.flatMap(g => g.fields).find(([n]) => n === field) || [field, field])[1];
      throw new Error(`Trường "${label}" (${field}) ${firstInvalid.reason}.`);
    }
    return payload;
  }

  function fillSample(kind) {
    const data = SAMPLES[kind];
    if (!data) return;
    for (const [k, v] of Object.entries(data)) {
      const el = $(k);
      if (el) {
        el.value = v;
        el.classList.remove("is-invalid");
      }
    }
  }

  /* ---------- Trạng thái kết quả ---------- */
  function clearResult() {
    $("resultEmpty").hidden = false;
    $("resultBody").hidden = true;
    $("resultBody").innerHTML = "";
    $("resultError").hidden = true;
    $("resultError").innerHTML = "";
  }

  function showError(message) {
    $("resultEmpty").hidden = true;
    $("resultBody").hidden = true;
    const box = $("resultError");
    box.hidden = false;
    box.innerHTML = `<div class="alert"><strong>❌ Không thể phân tích</strong>${message}</div>`;
  }

  function showResult(data) {
    const info = LABELS[data.class_id] ?? { en: data.prediction, vi: data.prediction, icon: "•", cls: "benign" };
    const time = new Date().toLocaleTimeString("vi-VN");

    $("resultEmpty").hidden = true;
    $("resultError").hidden = true;

    /* Probability bar (nếu có) */
    let probHTML = "";
    if (Array.isArray(data.probabilities) && data.probabilities.length === 2) {
      const pMal = data.probabilities[0] * 100;
      const pBen = data.probabilities[1] * 100;
      probHTML = `
        <div class="prob">
          <div class="prob__head"><span>⚠️ Ác tính</span><span>${pMal.toFixed(2)}%</span></div>
          <div class="prob__track"><div class="prob__fill prob__fill--danger" style="width:${pMal}%"></div></div>
        </div>
        <div class="prob">
          <div class="prob__head"><span>✅ Lành tính</span><span>${pBen.toFixed(2)}%</span></div>
          <div class="prob__track"><div class="prob__fill prob__fill--success" style="width:${pBen}%"></div></div>
        </div>
      `;
    }

    const body = $("resultBody");
    body.hidden = false;
    body.innerHTML = `
      <div class="result__card">
        <div class="verdict verdict--${info.cls}">
          <div class="verdict__icon">${info.icon}</div>
          <div class="verdict__label">${info.vi}</div>
          <div class="verdict__sub">${info.en}</div>
        </div>
        ${probHTML}
        <div class="meta">
          <div class="meta__row">
            <span class="meta__key">Mã lớp</span>
            <span class="meta__val">${data.class_id}</span>
          </div>
          <div class="meta__row">
            <span class="meta__key">Nhãn</span>
            <span class="meta__val">${data.prediction}</span>
          </div>
          <div class="meta__row">
            <span class="meta__key">Số đặc trưng</span>
            <span class="meta__val">30</span>
          </div>
          <div class="meta__row">
            <span class="meta__key">Thời điểm</span>
            <span class="meta__val">${time}</span>
          </div>
        </div>
        <div class="note">
          Kết quả chỉ mang tính tham khảo từ mô hình học máy,
          không thay thế kết luận của bác sĩ chuyên khoa.
        </div>
      </div>
    `;
  }

  /* ---------- Loading / Status ---------- */
  function setLoading(on) {
    $("btnSubmit").disabled = on;
    $("spinner").hidden = !on;
    $("btnText").textContent = on ? "Đang xử lý…" : "🔬 Phân tích";
  }

  function setStatus(state, extra = "") {
    const el = $("status");
    el.className = "status";
    if (state === "online")        { el.classList.add("is-online");   el.textContent = "API sẵn sàng"; }
    else if (state === "offline")  { el.classList.add("is-offline");  el.textContent = "Mất kết nối"; }
    else if (state === "checking") { el.classList.add("is-checking"); el.textContent = "Đang kiểm tra…"; }
    else                           { el.textContent = "Chưa kiểm tra"; }
    if (extra) el.title = extra;
  }

  /* ---------- Theme ---------- */
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    $("btnTheme").textContent = theme === "dark" ? "☀️" : "🌙";
    try { localStorage.setItem("bc-theme", theme); } catch {}
  }

  return {
    buildForm, readForm, fillSample, clearResult,
    showError, showResult, setLoading, setStatus, applyTheme,
  };
})();


/* =====================================================================
   MODULE 4 · CONTROLLER
   ===================================================================== */
const App = (() => {

  async function handlePredict(event) {
    event.preventDefault();

    let payload;
    try {
      payload = UI.readForm();
    } catch (err) {
      UI.showError(err.message);
      return;
    }

    UI.setLoading(true);
    try {
      const data = await Api.predict(payload);
      UI.showResult(data);
      UI.setStatus("online");
    } catch (err) {
      const hint = err.name === "AbortError"
        ? "Yêu cầu quá thời gian (timeout)."
        : err.message;
      UI.showError(`${hint}<br><br>Kiểm tra lại: FastAPI đang chạy? Model đã load chưa?`);
      UI.setStatus("offline", hint);
    } finally {
      UI.setLoading(false);
    }
  }

  async function checkHealth() {
    UI.setStatus("checking");
    try {
      const h = await Api.health();
      if (h && h.model_loaded) UI.setStatus("online");
      else UI.setStatus("offline", h && h.error ? h.error : "Model chưa sẵn sàng.");
    } catch {
      UI.setStatus("offline");
    }
  }

  function bindEvents() {
    document.getElementById("form").addEventListener("submit", handlePredict);

    document.getElementById("btnSampleMalignant").addEventListener("click", () => {
      UI.fillSample("malignant");
      UI.clearResult();
    });
    document.getElementById("btnSampleBenign").addEventListener("click", () => {
      UI.fillSample("benign");
      UI.clearResult();
    });

    document.getElementById("btnReset").addEventListener("click", () => {
      // form reset chạy trước → dùng setTimeout để chắc chắn
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
    // Khôi phục theme
    let saved = "light";
    try { saved = localStorage.getItem("bc-theme") || "light"; } catch {}
    UI.applyTheme(saved);

    UI.buildForm();
    bindEvents();
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
