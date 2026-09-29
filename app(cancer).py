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