# JOB CLASSIFICATION - PHIEN BAN DON GIAN
# Muc tieu: du doan career_level tu thong tin bai dang tuyen dung

# 1. IMPORT THU VIEN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTEN

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay
)


# 2. DOC DU LIEU
data = pd.read_excel("final_project.ods", engine="odf")

print("Kich thuoc du lieu:", data.shape)
print("\nThong tin du lieu:")
print(data.info())

print("\nGia tri thieu:")
print(data.isnull().sum())

print("\nSo dong trung lap:", data.duplicated().sum())


# 3. KHAM PHA NHAN (TARGET)
# print("\nSo luong tung career_level:")
# print(data["career_level"].value_counts())
#
# plt.figure(figsize=(10, 5))
# sns.countplot(
#     data=data, y="career_level",
#     order=data["career_level"].value_counts().index
# )
# plt.title("Phan bo career_level")
# plt.tight_layout()
# plt.show()


# 5. LAM SACH DU LIEU
data = data.drop_duplicates()
data = data.dropna(axis=0)

# 6. CHON FEATURE VA TARGET
features = [
    "title", "description",
    "function", "industry",
]
x = data[features].astype(str)
y = data["career_level"]

# 7. CHIA TRAIN VA TEST
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

#Xử lý mất cân bằng
ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy={
    "director_business_unit_leader": 500,
    "specialist": 500,
    "managing_director_small_medium_company": 500,
    "bereichsleiter": 1000
})

x_train, y_train = ros.fit_resample(x_train, y_train)


# 8. TIEN XU LY FEATURE
preprocessor = ColumnTransformer(
    transformers=[
        ("title_text", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
        ("description_text", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
        ("categories", OneHotEncoder(handle_unknown="ignore"), ["function", "industry"])
    ]
)


# 9. HUAN LUYEN VA SO SANH 2 MO HINH
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced'
    ),
    "Linear SVM": LinearSVC(class_weight='balanced')
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print(classification_report(y_test, predictions, zero_division=0))

    # ConfusionMatrixDisplay.from_predictions(
    #     y_test, predictions, xticks_rotation=45, cmap="Blues"
    # )
    # plt.title(f"Confusion Matrix - {name}")
    # plt.tight_layout()
    # plt.show()


# 10. BANG SO SANH KET QUA
results_df = pd.DataFrame(results).sort_values(
    by="Macro F1", ascending=False
)
print("\n===== SO SANH MO HINH =====")
print(results_df.to_string(index=False))

# ===== SO SANH MO HINH =====
#               Model  Accuracy  Macro F1  Weighted F1
# Logistic Regression  0.765363  0.695404     0.768044
#          Linear SVM  0.754811  0.497370     0.751637
