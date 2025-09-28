# app.py
import streamlit as st
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------
# 1. Load dataset (cached)
# -----------------------
@st.cache_resource
def load_pubmed_data():
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:1000]")

    def safe_text(field):
        if field is None:
            return ""
        if isinstance(field, list):
            return " ".join([str(x) for x in field])
        return str(field)

    abstracts = [safe_text(item['context']) for item in dataset]
    long_answers = [safe_text(item['long_answer']) for item in dataset]
    questions = [item['question'] for item in dataset]

    retrieval_contexts = [a + " " + l for a, l in zip(abstracts, long_answers)]
    return retrieval_contexts


# -----------------------
# 2. Embedding + FAISS
# -----------------------
@st.cache_resource
def build_index(retrieval_contexts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("gsarti/scibert-nli", device=device)

    embeddings = embed_model.encode(
        retrieval_contexts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=16
    )

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.cpu().numpy())

    return embed_model, index


# -----------------------
# 3. Load QA model
# -----------------------
@st.cache_resource
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model


# -----------------------
# 4. QA Function
# -----------------------
def answer_question(question, top_k=3):
    question_emb = embed_model.encode([question], convert_to_tensor=True)
    D, I = index.search(question_emb.cpu().numpy(), top_k)
    retrieved_contexts = " ".join([retrieval_contexts[i] for i in I[0]])

    input_text = f"answer the question: {question} based on context: {retrieved_contexts}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(inputs, max_length=150, num_beams=5)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    evidence = [retrieval_contexts[i] for i in I[0]]
    return answer, evidence


# -----------------------
# Streamlit UI
# -----------------------
st.title("🧪 PubMed QA Explorer")

retrieval_contexts = load_pubmed_data()
embed_model, index = build_index(retrieval_contexts)
tokenizer, model = load_qa_model()

user_question = st.text_input("Enter your biomedical question:")
if st.button("Get Answer") and user_question:
    answer, evidence = answer_question(user_question, top_k=3)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Supporting Evidence")
    for i, ctx in enumerate(evidence):
        st.markdown(f"**{i+1}.** {ctx[:300]}...")

