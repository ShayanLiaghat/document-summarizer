import streamlit as st
import logging
# import time
from transformers import pipeline
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO)

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        if not text.strip():
            raise ValueError("The uploaded PDF has no readable text.")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")
    
# Function to summarize text
def summarize_text(text, model, max_length=100, min_length=30):
    summarizer = pipeline("summarization", model=model)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

MAX_INPUT_LENGTH = 1024

def main():
    st.title("PDF Summarizer")
    st.write("Upload a PDF file, and I will summarize its content for you.")
    
    values = st.sidebar.slider("Select a range for summary length", 50, 200, (75, 100))
    model = st.sidebar.selectbox("Choose a summarization model", ["t5-small", "facebook/bart-large-cnn"])

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            # Extract text
            try:
                text = extract_text_from_pdf(uploaded_file)
                if len(text.split()) > MAX_INPUT_LENGTH:
                    st.warning(f"The input text exceeds {MAX_INPUT_LENGTH} words. Truncating to fit the model's limits.")
                    text = " ".join(text.split()[:MAX_INPUT_LENGTH])
                st.success("Text extraction complete!")
            except RuntimeError as e:
                st.error(str(e))
                return
        
        st.write("### Extracted Text:")
        st.text_area("Extracted Text", text, height=300)
        
    # Summarize text
    if st.button("Summarize"):
        with st.spinner("Summarizing text..."):
            summary = summarize_text(text, model, max_length=values[1], min_length=values[0])
            st.success("Summarization complete!")
            st.write("### Summary:")
            st.write(summary)

if __name__ == "__main__":
    main()