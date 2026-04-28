# =========================================
# GOJEK SENTIMENT ANALYSIS (FAST VERSION)
# =========================================

# ---------- 1. IMPORT ----------
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ---------- 2. LOAD DATA ----------
FILE_PATH = "C:/Users/USER/Documents/KULIAH/SEMESTER 4/RISET TEKNOLOGI INFORMASI/GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv"

df = pd.read_csv(FILE_PATH, encoding='utf-8')

print("Kolom:", df.columns)


# ---------- 3. DETECT KOLOM ----------
TEXT_COLUMN = None
for col in ["content", "review", "text"]:
    if col in df.columns:
        TEXT_COLUMN = col
        break

if TEXT_COLUMN is None:
    raise ValueError("Kolom teks tidak ditemukan!")

# label
if "sentiment" in df.columns:
    LABEL_COLUMN = "sentiment"

elif "score" in df.columns:
    def convert_rating(r):
        if r >= 4:
            return "positive"
        elif r <= 2:
            return "negative"
        else:
            return "neutral"

    df["sentiment"] = df["score"].apply(convert_rating)
    LABEL_COLUMN = "sentiment"

else:
    raise ValueError("Kolom label tidak ditemukan!")


print("Text:", TEXT_COLUMN)
print("Label:", LABEL_COLUMN)
print("\nDistribusi:\n", df[LABEL_COLUMN].value_counts())


# ---------- 4. CLEAN TEXT ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

df = df[df["clean_text"].str.len() > 2]


# ---------- 5. SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df[LABEL_COLUMN],
    test_size=0.2,
    stratify=df[LABEL_COLUMN],
    random_state=42
)


# ---------- 6. FAST MODEL ----------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        max_df=0.9,
        min_df=2,
        max_features=10000   # penting biar cepat
    )),
    ("svm", LinearSVC(C=1))
])


# ---------- 7. TRAIN ----------
print("\nTraining cepat...")
model.fit(X_train, y_train)


# ---------- 8. EVALUASI ----------
y_pred = model.predict(X_test)

print("\n=== HASIL ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ---------- 9. SAVE ----------
import joblib
joblib.dump(model, "gojek_fast_model.pkl")

print("\nModel disimpan!")

# ---------- 10. VISUALISASI ----------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- 1. CONFUSION MATRIX ---
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

plt.figure()
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# --- 2. DISTRIBUSI LABEL ASLI ---
plt.figure()
df[LABEL_COLUMN].value_counts().plot(kind='bar')
plt.title("Distribusi Label (Dataset)")
plt.xlabel("Sentiment")
plt.ylabel("Jumlah")
plt.xticks(rotation=0)
plt.show()


# --- 3. DISTRIBUSI HASIL PREDIKSI ---
import pandas as pd

pred_series = pd.Series(y_pred)

plt.figure()
pred_series.value_counts().plot(kind='bar')
plt.title("Distribusi Hasil Prediksi")
plt.xlabel("Sentiment")
plt.ylabel("Jumlah")
plt.xticks(rotation=0)
plt.show()