from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load mô hình SVM đã huấn luyện
model = joblib.load("svm_model.pkl")

app = FastAPI(
    title="Breast Cancer Classification API",
    description="SVM model for the Breast Cancer dataset",
    version="1.0.0",
)


# 30 đặc trưng của load_breast_cancer()
class BreastCancerInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


# Nhãn của dữ liệu Breast Cancer
cancer_classes = {
    0: "malignant",
    1: "benign",
}


@app.get("/")
def home():
    return {
        "message": "Breast Cancer SVM API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: BreastCancerInput):

    features = [[
        data.mean_radius,
        data.mean_texture,
        data.mean_perimeter,
        data.mean_area,
        data.mean_smoothness,
        data.mean_compactness,
        data.mean_concavity,
        data.mean_concave_points,
        data.mean_symmetry,
        data.mean_fractal_dimension,

        data.radius_error,
        data.texture_error,
        data.perimeter_error,
        data.area_error,
        data.smoothness_error,
        data.compactness_error,
        data.concavity_error,
        data.concave_points_error,
        data.symmetry_error,
        data.fractal_dimension_error,

        data.worst_radius,
        data.worst_texture,
        data.worst_perimeter,
        data.worst_area,
        data.worst_smoothness,
        data.worst_compactness,
        data.worst_concavity,
        data.worst_concave_points,
        data.worst_symmetry,
        data.worst_fractal_dimension,
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "class_id": prediction,
        "prediction": cancer_classes[prediction],
    }



<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Breast Cancer · SVM Diagnostic</title>

<style>
/* ============================================================
   1. DESIGN TOKENS  (single source of truth)
   ============================================================ */
:root {
  --bg:        #eef2f8;
  --surface:   #ffffff;
  --surface-2: #f8fafc;
  --text:      #0f172a;
  --muted:     #64748b;
  --border:    #e2e8f0;
  --border-2:  #cbd5e1;

  --primary:      #2563eb;
  --primary-dark: #1d4ed8;
  --primary-soft: #dbeafe;

  --danger:      #dc2626;
  --danger-soft: #fef2f2;
  --danger-bd:   #fecaca;

  --success:      #16a34a;
  --success-soft: #f0fdf4;
  --success-bd:   #bbf7d0;

  --radius:    14px;
  --radius-sm: 10px;
  --radius-xs: 8px;

  --shadow-sm: 0 1px 2px rgba(15, 23, 42, .06);
  --shadow:    0 1px 2px rgba(15, 23, 42, .06), 0 8px 24px rgba(15, 23, 42, .06);
  --shadow-lg: 0 12px 34px rgba(15, 23, 42, .12);

  --font: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
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
  line-height: 1.5;
  color: var(--text);
  background:
    radial-gradient(1100px 520px at 6% -10%, #dbeafe 0%, transparent 60%),
    radial-gradient(900px 480px at 100% 0%, #ede9fe 0%, transparent 55%),
    var(--bg);
  padding: 24px 16px 64px;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}
h1, h2, h3, p { margin: 0; }
button, input { font: inherit; color: inherit; }

/* ============================================================
   3. LAYOUT
   ============================================================ */
.app {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 20px;
  align-items: start;
}
@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; }
}

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
  border-radius: var(--radius);
  padding: 18px 22px;
  box-shadow: var(--shadow);
}
.brand { display: flex; align-items: center; gap: 14px; }
.brand__icon {
  width: 46px; height: 46px; flex: 0 0 46px;
  display: grid; place-items: center;
  border-radius: 13px;
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  color: #fff;
  font-size: 22px;
  box-shadow: 0 6px 16px rgba(37, 99, 235, .35);
}
.brand__title { font-size: 19px; font-weight: 700; letter-spacing: -.2px; }
.brand__sub   { font-size: 13px; color: var(--muted); margin-top: 2px; }

.status {
  font-size: 12.5px;
  font-weight: 600;
  padding: 7px 13px;
  border-radius: 999px;
  background: #f1f5f9;
  color: var(--muted);
  border: 1px solid var(--border);
  white-space: nowrap;
  transition: all .25s;
}
.status.is-online  { background: var(--success-soft); color: var(--success); border-color: var(--success-bd); }
.status.is-offline { background: var(--danger-soft);  color: var(--danger);  border-color: var(--danger-bd); }
.status.is-checking{ background: #fffbeb; color: #b45309; border-color: #fde68a; }

/* ============================================================
   5. PANELS
   ============================================================ */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.panel__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 16px 22px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #ffffff, #fbfcfe);
}
.panel__head h2 { font-size: 15px; font-weight: 700; }
.panel__body { padding: 20px 22px; }

/* ============================================================
   6. FEATURE GROUPS
   ============================================================ */
.groups {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 16px;
  padding: 20px 22px;
}
.group {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px;
  background: var(--surface-2);
  transition: border-color .15s;
}
.group:hover { border-color: var(--border-2); }

.group__title {
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: var(--primary);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 7px;
}
.group__title::before {
  content: "";
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-soft);
}

.fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(124px, 1fr));
  gap: 10px;
}
.field label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  margin-bottom: 5px;
  letter-spacing: .02em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.field input {
  width: 100%;
  padding: 9px 11px;
  font-size: 13.5px;
  font-family: var(--mono);
  background: #fff;
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  transition: border-color .15s, box-shadow .15s, background .15s;
}
.field input:hover { border-color: var(--border-2); }
.field input:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, .14);
}
.field input:invalid:not(:placeholder-shown) {
  border-color: var(--danger);
  background: #fff5f5;
}
/* hide number spinners for a cleaner look */
.field input::-webkit-outer-spin-button,
.field input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.field input[type=number] { -moz-appearance: textfield; }

/* ============================================================
   7. BUTTONS
   ============================================================ */
.actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  padding: 18px 22px;
  border-top: 1px solid var(--border);
  background: var(--surface-2);
}
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 11px 22px;
  font-size: 14px;
  font-weight: 600;
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
  cursor: pointer;
  user-select: none;
  transition: transform .12s, box-shadow .15s, background .15s, color .15s;
}
.btn:active:not(:disabled) { transform: translateY(1px); }
.btn:disabled { opacity: .6; cursor: not-allowed; }

.btn--primary {
  background: linear-gradient(135deg, #2563eb, #4f46e5);
  color: #fff;
  box-shadow: 0 6px 16px rgba(37, 99, 235, .3);
}
.btn--primary:hover:not(:disabled) { box-shadow: 0 8px 22px rgba(37, 99, 235, .42); }

.btn--ghost {
  background: #fff;
  color: var(--muted);
  border-color: var(--border);
}
.btn--ghost:hover:not(:disabled) { background: #f1f5f9; color: var(--text); }

.btn--sm { padding: 7px 14px; font-size: 12.5px; }

.spinner {
  width: 15px; height: 15px;
  border: 2px solid rgba(255, 255, 255, .35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ============================================================
   8. RESULT PANEL
   ============================================================ */
.result { position: sticky; top: 24px; }

.result__empty {
  padding: 48px 22px;
  text-align: center;
  color: var(--muted);
}
.result__empty .icon {
  font-size: 40px;
  opacity: .3;
  display: block;
  margin-bottom: 12px;
}
.result__empty p { font-size: 13.5px; line-height: 1.65; }

.result__card { padding: 22px; animation: fadeUp .3s ease; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: none; }
}

.verdict {
  border-radius: var(--radius-sm);
  padding: 22px 16px;
  text-align: center;
  margin-bottom: 18px;
}
.verdict--malignant { background: var(--danger-soft);  border: 1px solid var(--danger-bd); }
.verdict--benign    { background: var(--success-soft); border: 1px solid var(--success-bd); }
.verdict__icon  { font-size: 36px; line-height: 1; margin-bottom: 10px; }
.verdict__label { font-size: 19px; font-weight: 800; letter-spacing: .02em; text-transform: uppercase; }
.verdict--malignant .verdict__label { color: var(--danger); }
.verdict--benign    .verdict__label { color: var(--success); }
.verdict__sub   { font-size: 12.5px; color: var(--muted); margin-top: 6px; letter-spacing: .04em; }

.meta { display: flex; flex-direction: column; }
.meta__row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 11px 0;
  font-size: 13.5px;
  border-bottom: 1px dashed var(--border);
}
.meta__row:last-child { border-bottom: none; }
.meta__key { color: var(--muted); }
.meta__val { font-weight: 600; font-family: var(--mono); }

.alert {
  margin: 0 22px 20px;
  padding: 14px 16px;
  border-radius: var(--radius-sm);
  background: var(--danger-soft);
  border: 1px solid var(--danger-bd);
  color: #991b1b;
  font-size: 13px;
  line-height: 1.6;
  animation: fadeUp .25s ease;
}
.alert strong { display: block; margin-bottom: 4px; }

.note {
  margin-top: 16px;
  padding: 12px 14px;
  font-size: 11.5px;
  line-height: 1.65;
  color: var(--muted);
  background: var(--surface-2);
  border-left: 3px solid var(--border-2);
  border-radius: 0 8px 8px 0;
}

/* ============================================================
   9. FOOTER
   ============================================================ */
.footer {
  text-align: center;
  font-size: 12px;
  color: var(--muted);
  padding-top: 4px;
}

/* ============================================================
   10. PRINT
   ============================================================ */
@media print {
  body { background: #fff; padding: 0; }
  .header, .actions, .footer, #btnSample, #btnReset { display: none !important; }
  .result { position: static; }
}
</style>
</head>

<body>
<div class="app">

  <!-- ===================== HEADER ===================== -->
  <header class="header">
    <div class="brand">
      <div class="brand__icon" aria-hidden="true">⚕</div>
      <div>
        <h1 class="brand__title">Chẩn đoán ung thư vú</h1>
        <p class="brand__sub">Mô hình SVM · 30 đặc trưng tế bào (FNA)</p>
      </div>
    </div>
    <span class="status" id="status">● Đang kiểm tra…</span>
  </header>

  <!-- ===================== MAIN ===================== -->
  <div class="layout">

    <!-- ---------- FORM ---------- -->
    <form class="panel" id="form" autocomplete="off" novalidate>
      <div class="panel__head">
        <h2>Thông số xét nghiệm</h2>
        <button type="button" class="btn btn--ghost btn--sm" id="btnSample">
          Điền mẫu
        </button>
      </div>

      <div class="groups" id="groups"><!-- sinh bằng JS --></div>

      <div class="actions">
        <button type="submit" class="btn btn--primary" id="btnSubmit">
          <span class="spinner" id="spinner" hidden></span>
          <span id="btnText">Phân tích</span>
        </button>
        <button type="reset" class="btn btn--ghost" id="btnReset">Xoá tất cả</button>
      </div>
    </form>

    <!-- ---------- RESULT ---------- -->
    <aside class="panel result" id="result">
      <div class="panel__head"><h2>Kết quả</h2></div>

      <div class="result__empty" id="resultEmpty">
        <span class="icon" aria-hidden="true">🔬</span>
        <p>Nhập đầy đủ 30 chỉ số<br />rồi bấm <strong>Phân tích</strong>.</p>
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
  API_URL:    "http://127.0.0.1:8000/predict",
  HEALTH_URL: "http://127.0.0.1:8000/health",
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

/* Một ca malignant điển hình trong load_breast_cancer() */
const SAMPLE = {
  mean_radius: 17.99, mean_texture: 10.38, mean_perimeter: 122.8, mean_area: 1001.0,
  mean_smoothness: 0.1184, mean_compactness: 0.2776, mean_concavity: 0.3001,
  mean_concave_points: 0.1471, mean_symmetry: 0.2419, mean_fractal_dimension: 0.07871,

  radius_error: 1.095, texture_error: 0.9053, perimeter_error: 8.589, area_error: 153.4,
  smoothness_error: 0.006399, compactness_error: 0.04904, concavity_error: 0.05373,
  concave_points_error: 0.01587, symmetry_error: 0.03003, fractal_dimension_error: 0.006193,

  worst_radius: 25.38, worst_texture: 17.33, worst_perimeter: 184.6, worst_area: 2019.0,
  worst_smoothness: 0.1622, worst_compactness: 0.6656, worst_concavity: 0.7119,
  worst_concave_points: 0.2654, worst_symmetry: 0.4601, worst_fractal_dimension: 0.1189,
};

const LABELS = {
  0: { en: "Malignant", vi: "Ác tính",   icon: "⚠️", cls: "malignant" },
  1: { en: "Benign",    vi: "Lành tính", icon: "✅", cls: "benign"    },
};


/* =====================================================================
   MODULE 2 · API
   Tách hoàn toàn khỏi UI — dễ test / thay thế.
   ===================================================================== */
const Api = (() => {
  async function request(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} — ${detail || res.statusText}`);
      }
      return await res.json();
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
    health() {
      return request(CONFIG.HEALTH_URL);
    },
  };
})();


/* =====================================================================
   MODULE 3 · UI
   Chỉ thao tác DOM, không chứa logic nghiệp vụ.
   ===================================================================== */
const UI = (() => {
  const $ = (id) => document.getElementById(id);

  /* ---------- Sinh form từ CONFIG ---------- */
  function buildForm() {
    const html = FEATURE_GROUPS.map((g) => `
      <section class="group">
        <h3 class="group__title">${g.title}</h3>
        <div class="fields">
          ${g.fields.map(([name, label]) => `
            <div class="field">
              <label for="${name}" title="${label}">${label}</label>
              <input id="${name}" name="${name}" type="number"
                     step="any" min="0" placeholder="0.0" required />
            </div>
          `).join("")}
        </div>
      </section>
    `).join("");
    $("groups").innerHTML = html;
  }

  /* ---------- Đọc & validate dữ liệu form ---------- */
  function readForm() {
    const payload = {};
    for (const g of FEATURE_GROUPS) {
      for (const [name] of g.fields) {
        const el = $(name);
        const raw = el.value.trim();
        if (raw === "") throw new Error(`Vui lòng nhập giá trị cho "${name}".`);
        const val = Number(raw);
        if (!Number.isFinite(val)) throw new Error(`Giá trị không hợp lệ cho "${name}".`);
        if (val < 0) throw new Error(`"${name}" không được âm.`);
        payload[name] = val;
      }
    }
    return payload;
  }

  function fillSample() {
    for (const [k, v] of Object.entries(SAMPLE)) {
      const el = $(k);
      if (el) el.value = v;
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
    box.innerHTML = `<div class="alert"><strong>Không thể phân tích</strong>${message}</div>`;
  }

  function showResult(data) {
    const info = LABELS[data.class_id] ?? {
      en: data.prediction, vi: data.prediction, icon: "•", cls: "benign",
    };
    const time = new Date().toLocaleTimeString("vi-VN");

    $("resultEmpty").hidden = true;
    $("resultError").hidden = true;

    const body = $("resultBody");
    body.hidden = false;
    body.innerHTML = `
      <div class="result__card">
        <div class="verdict verdict--${info.cls}">
          <div class="verdict__icon">${info.icon}</div>
          <div class="verdict__label">${info.vi}</div>
          <div class="verdict__sub">${info.en}</div>
        </div>

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

  /* ---------- Nút / trạng thái ---------- */
  function setLoading(on) {
    $("btnSubmit").disabled = on;
    $("spinner").hidden = !on;
    $("btnText").textContent = on ? "Đang xử lý…" : "Phân tích";
  }

  function setStatus(state) {
    const el = $("status");
    el.className = "status";
    if (state === "online")        { el.classList.add("is-online");   el.textContent = "● API sẵn sàng"; }
    else if (state === "offline")  { el.classList.add("is-offline");  el.textContent = "● Mất kết nối"; }
    else if (state === "checking") { el.classList.add("is-checking"); el.textContent = "● Đang kiểm tra…"; }
    else                           { el.textContent = "● Chưa kiểm tra"; }
  }

  return { buildForm, readForm, fillSample, clearResult, showError, showResult, setLoading, setStatus };
})();


/* =====================================================================
   MODULE 4 · CONTROLLER
   Kết nối UI ↔ API, điều phối luồng.
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
      UI.showError(
        `${hint}<br><br>Kiểm tra lại: FastAPI đang chạy? Đã bật CORS?`
      );
      UI.setStatus("offline");
    } finally {
      UI.setLoading(false);
    }
  }

  async function checkHealth() {
    UI.setStatus("checking");
    try {
      await Api.health();
      UI.setStatus("online");
    } catch {
      UI.setStatus("offline");
    }
  }

  function bindEvents() {
    document.getElementById("form").addEventListener("submit", handlePredict);

    document.getElementById("btnSample").addEventListener("click", () => {
      UI.fillSample();
      UI.clearResult();
    });

    document.getElementById("btnReset").addEventListener("click", () => {
      // form reset mặc định chạy trước → dùng setTimeout để chắc chắn
      setTimeout(() => UI.clearResult(), 0);
    });
  }

  function init() {
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
