from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import joblib

breast_cancer = load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

model = SVC(kernel="linear")
model.fit(X, y)

joblib.dump(model, "svm_model.pkl")

print("Model saved!")