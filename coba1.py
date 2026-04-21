# =========================================
# SENTIMENT ANALYSIS SHOPEE (FINAL FIX + SAVING FIGURES)
# =========================================

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# =========================================
# 1. LOAD DATA
# =========================================
file_path = "C:/Users/USER/Documents/KULIAH/SEMESTER 4/RISET TEKNOLOGI INFORMASI/Shopee_Sampled_Reviews.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()
print("Kolom dataset:", df.columns)

# =========================================
# 2. KOLOM YANG DIGUNAKAN
# =========================================
text_column = 'content'
label_column = 'sentimen'

# =========================================
# 3. LABEL DARI SCORE
# =========================================
def convert_rating_to_sentiment(score):
    if score >= 4:
        return 'positif'
    else:
        return 'negatif'

df[label_column] = df['score'].apply(convert_rating_to_sentiment)

# =========================================
# 4. PREPROCESSING
# =========================================
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df[text_column] = df[text_column].apply(clean_text)

# =========================================
# 5. VISUALISASI DISTRIBUSI
# =========================================
plt.figure()
df[label_column].value_counts().plot(kind='bar')
plt.title("Distribusi Sentimen Shopee")
plt.savefig("Distribusi_Sentimen.png", dpi=300, bbox_inches='tight')  # Save figure
plt.show()

# =========================================
# 6. TF-IDF
# =========================================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df[text_column])
y = df[label_column]

# =========================================
# 7. SPLIT DATA
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 8. MODEL SVM
# =========================================
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# =========================================
# 9. PREDIKSI
# =========================================
y_pred = model.predict(X_test)

# =========================================
# 10. EVALUASI
# =========================================
accuracy = accuracy_score(y_test, y_pred)
print("\n=== HASIL ===")
print("Akurasi: {:.2f}%".format(accuracy * 100))
print(classification_report(y_test, y_pred))

# =========================================
# 11. CONFUSION MATRIX
# =========================================
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("Confusion_Matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================
# 12. GRAFIK AKURASI
# =========================================
plt.figure()
plt.bar(["Akurasi"], [accuracy])
plt.title("Akurasi Model")
plt.ylim(0,1)
plt.savefig("Akurasi_Model.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================
# 13. TOP KATA TF-IDF
# =========================================
feature_names = tfidf.get_feature_names_out()
tfidf_scores = X_train.mean(axis=0).A1

top_n = 10
top_indices = tfidf_scores.argsort()[-top_n:]
top_words = [feature_names[i] for i in top_indices]
top_values = tfidf_scores[top_indices]

plt.figure()
plt.barh(top_words, top_values)
plt.title("Top Kata TF-IDF")
plt.savefig("Top_Kata_TFIDF.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================
# 14. PANJANG ULASAN
# =========================================
df['review_length'] = df[text_column].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,4))
sns.histplot(df, x='review_length', hue=label_column, bins=30, palette='coolwarm', kde=True)
plt.title('Distribusi Panjang Ulasan per Sentimen')
plt.xlabel('Jumlah Kata')
plt.ylabel('Jumlah Ulasan')
plt.savefig("Panjang_Ulasan.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================
# 15. RANGKUMAN METRIK PER KELAS
# =========================================
report_dict = classification_report(y_test, y_pred, output_dict=True)
labels = df[label_column].unique()
precision = [report_dict[l]['precision'] for l in labels]
recall = [report_dict[l]['recall'] for l in labels]
f1 = [report_dict[l]['f1-score'] for l in labels]

x = np.arange(len(labels))
width = 0.2

plt.figure(figsize=(10,6))
plt.bar(x-width, precision, width, label='Precision', color='skyblue')
plt.bar(x, recall, width, label='Recall', color='lightgreen')
plt.bar(x+width, f1, width, label='F1-score', color='salmon')
plt.xticks(x, labels)
plt.ylim(0,1)
plt.title('Rangkuman Metrik per Sentimen')
plt.ylabel('Skor')
plt.legend()
plt.savefig("Rangkuman_Metrik.png", dpi=300, bbox_inches='tight')
plt.show()
