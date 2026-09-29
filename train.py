from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import joblib

# Load dữ liệu Breast Cancer
data = load_breast_cancer()

X = data.data
y = data.target

# Tạo mô hình SVM
model = SVC(kernel="linear")

# Huấn luyện
model.fit(X, y)

# Lưu mô hình
joblib.dump(model, "svm_model.pkl")

print("Model saved!")