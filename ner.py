from transformers import pipeline
import pandas as pd

# Inisialisasi model NER Bahasa Indonesia dari Hugging Face
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

# Load file CSV
file_path = "D:/project-experience/sentimen-analisis/ulasan_shopee (2).csv"
data = pd.read_csv(file_path)

# Pastikan kolom 'content' ada dan tidak kosong
if 'content' not in data.columns or data['content'].isnull().all():
    raise ValueError("Kolom 'content' tidak ditemukan atau semua datanya kosong.")

# Fungsi untuk NER
def extract_entities(text):
    entities = ner_pipeline(text)
    # Formatkan hasil NER
    formatted_entities = [{"entity": ent['entity'], "word": ent['word'], "score": ent['score']} for ent in entities]
    return formatted_entities

# Proses data untuk ekstraksi NER
ners = []

for content in data['content']:
    if pd.isna(content):  # Skip jika data kosong
        ners.append(None)
        continue

    # Ekstraksi entitas
    entities = extract_entities(content)
    ners.append(entities)

# Tambahkan hasil NER ke DataFrame
data['ner'] = ners

# Simpan hasil ke file baru
output_path = "output_with_ner.csv"
data.to_csv(output_path, index=False)

# Menampilkan ringkasan hasil
print(f"Proses selesai. Hasil disimpan di: {output_path}")
