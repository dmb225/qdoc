import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq


def execute_rag(llm: Groq, model_name: str) -> None:
    st.title("Q&A with Your Documents")
    st.markdown("Upload your `.pdf` files to build a semantic search index and ask questions.")
    uploaded_files = st.file_uploader("Upload text files", type="pdf", accept_multiple_files=True)
    query = st.text_input("Ask a question about the documents")
    if uploaded_files and query:
        with (
            st.spinner("Processing documents and building index..."),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            # Save uploaded files to a temporary directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Load documents
            documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()

            # Create vector index
            embedding_model = HuggingFaceEmbedding(
                model_name=model_name,
            )
            index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
            index.storage_context.persist()

            # Retrieve and query
            query_engine = index.as_query_engine(llm=llm)
            response = query_engine.query(query)

            # Display results
            st.subheader("Answer:")
            st.write(str(response))


def main() -> None:
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise OSError("Missing GROQ_API_KEY environment variable.")

    rag_llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)
    execute_rag(rag_llm, "sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()
