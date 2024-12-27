from transformers import pipeline
import pandas as pd

# Inisialisasi model sentiment analysis dari Hugging Face
pretrained_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=pretrained_name,
    tokenizer=pretrained_name
)

# Load file CSV
file_path = "D:/project-experience/sentimen-analisis/ulasan_shopee (2).csv"
data = pd.read_csv(file_path)

# Pastikan kolom 'content' ada dan tidak kosong
if 'content' not in data.columns or data['content'].isnull().all():
    raise ValueError("Kolom 'content' tidak ditemukan atau semua datanya kosong.")

# Fungsi untuk menganalisis sentimen
def analyze_sentiment(text):
    result = sentiment_pipeline(text[:512])  # Pastikan teks tidak melebihi panjang maksimum model
    label = result[0]['label']
    score = result[0]['score']
    return label, score

# Proses data untuk analisis sentimen
sentiments = []

for content in data['content']:
    if pd.isna(content):  # Skip jika data kosong
        sentiments.append({'sentiment': None, 'score': None})
        continue

    # Analisis sentimen
    label, score = analyze_sentiment(content)
    sentiments.append({'sentiment': label, 'score': score})

# Tambahkan hasil analisis ke DataFrame
data['sentiment'] = [s['sentiment'] for s in sentiments]
data['sentiment_score'] = [s['score'] for s in sentiments]

# Simpan hasil ke file baru
output_path = "output_with_sentiment.csv"
data.to_csv(output_path, index=False)

# Filter untuk sentimen positif dan negatif
positive_data = data[data['sentiment'] == 'positive']
negative_data = data[data['sentiment'] == 'negative']

# Menampilkan ringkasan hasil
print(f"Jumlah ulasan positif: {len(positive_data)}")
print(f"Jumlah ulasan negatif: {len(negative_data)}")
print(f"Hasil disimpan di: {output_path}")
