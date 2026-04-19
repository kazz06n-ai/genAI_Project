import streamlit as st
from brain import get_rag_chain

# Page Config
st.set_page_config(page_title="Cyber Threat Gen", page_icon="🛡️", layout="wide")

st.title("🛡️ GenAI Cybersecurity Threat Reporter")
st.markdown("---")

# Sidebar for Project Info (Great for BTech Demos)
with st.sidebar:
    st.header("Project Details")
    st.info("System: Local RAG\n\nModel: Ollama (gemma4)\n\nDatabase: ChromaDB")
    if st.button("Clear History"):
        st.rerun()

# Main Input Section
user_query = st.text_area(
    "Describe the threat or paste logs here:",
    placeholder="e.g., Multiple brute force attempts detected on port 22...",
    height=150
)

if st.button("Generate Professional Report"):
    if user_query:
        with st.spinner("🔍 Analyzing threat vectors and generating report..."):
            try:
                # Initialize the chain from brain.py
                rag_chain = get_rag_chain()

                # Get response
                response = rag_chain.invoke(user_query)

                # Display Results
                st.subheader("Generated Threat Report")
                st.markdown(response["result"])

                # Download Option
                st.download_button(
                    label="Download Report as Text",
                    data=response["result"],
                    file_name="threat_report.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}. Make sure Ollama is running!")
    else:
        st.warning("Please enter a threat description first.")

st.markdown("---")
st.caption("BTech CSE Project - AI-Powered Threat Analysis System")