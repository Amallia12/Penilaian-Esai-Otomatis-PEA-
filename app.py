import streamlit as st
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load IndoBERT
@st.cache_resource
def load_model():
    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model

# Function to get embeddings
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# Function to calculate similarity
def calculate_similarity(student_answer, reference_answer, tokenizer, model):
    student_embedding = get_bert_embedding(student_answer, tokenizer, model)
    reference_embedding = get_bert_embedding(reference_answer, tokenizer, model)
    similarity = cosine_similarity(student_embedding.numpy(), reference_embedding.numpy())[0][0]
    return similarity

# Streamlit App
st.title("Penilaian Esai Otomatis Menggunakan IndoBERT")
st.write("Masukkan jawaban siswa dan kunci jawaban untuk menghitung nilai kemiripan.")

# Input fields
key_answer = st.text_area("Kunci Jawaban", placeholder="Masukkan kunci jawaban di sini...")
student_answer = st.text_area("Jawaban Siswa", placeholder="Masukkan jawaban siswa di sini...")

# Load model
tokenizer, model = load_model()

# Process similarity
if st.button("Hitung Kesamaan"):
    if not key_answer or not student_answer:
        st.warning("Harap isi kedua kolom teks!")
    else:
        similarity_score = calculate_similarity(student_answer, key_answer, tokenizer, model)
        st.success(f"Nilai Kesamaan: {similarity_score:.4f}")
