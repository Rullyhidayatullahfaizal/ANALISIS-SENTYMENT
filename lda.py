import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import nltk

# Pastikan library berikut telah diinstal
# pip install pandas scikit-learn nltk openpyxl

# Download stopwords NLTK (hanya pertama kali)
nltk.download('stopwords')

# Baca file CSV
file_path = "output_with_sentiment.csv"  # File CSV hasil analisis sentimen
data = pd.read_csv(file_path)

# Pastikan kolom sentiment dan content ada
if 'content' not in data.columns:
    raise ValueError("Kolom 'content' tidak ditemukan dalam data.")

# Fungsi pembersihan teks
def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))  # Stopwords bahasa Indonesia
    text = re.sub(r'[^A-Za-z\s]', '', str(text))  # Hapus karakter non-huruf
    text = text.lower()  # Ubah teks ke huruf kecil
    words = [word for word in text.split() if word not in stop_words]  # Hapus stopwords
    return ' '.join(words)

# Bersihkan ulasan di kolom "content"
data['cleaned_content'] = data['content'].apply(clean_text)

# Analisis Sentimen Dummy (sesuaikan dengan model sentimen Anda)
# Labeling manual berdasarkan skor dummy (contoh saja):
# Positif jika ada kata tertentu, negatif jika tidak
data['sentiment'] = data['cleaned_content'].apply(lambda x: 'positive' if 'bagus' in x else 'negative')

# Pisahkan ulasan positif dan negatif berdasarkan label sentimen
positive_reviews_cleaned = data[data['sentiment'] == 'positive']['cleaned_content']
negative_reviews_cleaned = data[data['sentiment'] == 'negative']['cleaned_content']

# Fungsi untuk menampilkan topik
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(", ".join(top_words))
    return topics

# LSA dan LDA untuk kedua kategori (positif dan negatif)

# LSA (Latent Semantic Analysis) untuk Ulasan Positif
vectorizer_positive = TfidfVectorizer(max_features=1000)
X_positive = vectorizer_positive.fit_transform(positive_reviews_cleaned)
lsa_positive = TruncatedSVD(n_components=2, random_state=42)
lsa_positive.fit(X_positive)
feature_names_positive = vectorizer_positive.get_feature_names_out()
lsa_topics_positive = display_topics(lsa_positive, feature_names_positive, 10)

# LSA untuk Ulasan Negatif
vectorizer_negative = TfidfVectorizer(max_features=1000)
X_negative = vectorizer_negative.fit_transform(negative_reviews_cleaned)
lsa_negative = TruncatedSVD(n_components=2, random_state=42)
lsa_negative.fit(X_negative)
feature_names_negative = vectorizer_negative.get_feature_names_out()
lsa_topics_negative = display_topics(lsa_negative, feature_names_negative, 10)

# LDA (Latent Dirichlet Allocation) untuk Ulasan Positif
vectorizer_positive_lda = CountVectorizer(max_features=1000)
X_positive_lda = vectorizer_positive_lda.fit_transform(positive_reviews_cleaned)
lda_positive = LatentDirichletAllocation(n_components=2, random_state=42)
lda_positive.fit(X_positive_lda)
feature_names_positive_lda = vectorizer_positive_lda.get_feature_names_out()
lda_topics_positive = display_topics(lda_positive, feature_names_positive_lda, 10)

# LDA untuk Ulasan Negatif
vectorizer_negative_lda = CountVectorizer(max_features=1000)
X_negative_lda = vectorizer_negative_lda.fit_transform(negative_reviews_cleaned)
lda_negative = LatentDirichletAllocation(n_components=2, random_state=42)
lda_negative.fit(X_negative_lda)
feature_names_negative_lda = vectorizer_negative_lda.get_feature_names_out()
lda_topics_negative = display_topics(lda_negative, feature_names_negative_lda, 10)

# Tambahkan hasil ke DataFrame
data['lsa_topics_positive'] = ', '.join(lsa_topics_positive)
data['lsa_topics_negative'] = ', '.join(lsa_topics_negative)
data['lda_topics_positive'] = ', '.join(lda_topics_positive)
data['lda_topics_negative'] = ', '.join(lda_topics_negative)

# Simpan DataFrame ke file Excel
output_file = "output_analysis.xlsx"
data.to_excel(output_file, index=False)

print(f"Analisis sentimen dan topik telah disimpan ke file: {output_file}")
