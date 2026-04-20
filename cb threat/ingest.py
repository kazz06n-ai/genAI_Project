import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data/"
DB_PATH = "db/"

def create_vector_db():
    if os.path.exists(DB_PATH):
        print("🧹 Cleaning old database...")
        shutil.rmtree(DB_PATH)

    print("📂 Loading documents (PDF, TXT, CSV)...")
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    csv_loader = DirectoryLoader(DATA_PATH, glob="*.csv", loader_cls=CSVLoader)

    docs = pdf_loader.load() + txt_loader.load() + csv_loader.load()

    if not docs:
        print("❌ No documents found in 'data/' folder!")
        return

    print(f"✂️ Splitting {len(docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    print("🤖 Generating embeddings (CUDA Acceleration Enabled)...")
    # UPDATED: device set to 'cuda'
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )

    print(f"💾 Saving {len(chunks)} chunks to {DB_PATH}...")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_PATH)
    print("✅ Success! Vector Database Created on GPU.")


if __name__ == "__main__":
    create_vector_db()