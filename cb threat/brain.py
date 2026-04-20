from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def get_rag_chain():
    # UPDATED: Load embedding model on CUDA
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )

    vector_db = Chroma(persist_directory="./db", embedding_function=embeddings)

    # Ollama automatically uses your GPU if available.
    # Ensure your NVIDIA drivers and Toolkit are up to date.
    llm = Ollama(model="gemma4", temperature=0.1)

    template = """
    ROLE: You are a professional Cybersecurity AI.

    STRICT RULES:
    1. ONLY answer questions about cybersecurity, network logs, or threats.
    2. If the user asks for a joke, poem, or casual talk, reply: "I am a specialized Cybersecurity AI and am not programmed for entertainment or casual conversation."
    3. If the input is gibberish or random characters, reply: "Invalid Input. Please provide valid security logs or queries."
    4. Use the context below to build your report.

    CONTEXT: {context}
    USER INPUT: {question}

    REPORT FORMAT:
    ### 🛡️ Executive Summary
    ### 🔍 Technical Analysis (Timestamps, IPs, Attack Types)
    ### 🛠️ Mitigation Steps
    ### ⚠️ Risk Score (1-10)
    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return chain