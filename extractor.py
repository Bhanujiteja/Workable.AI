import os
import pdfplumber
import pytesseract
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import shutil
from groq import Groq
from dotenv import load_dotenv

# Load .env if available
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_latex_questions_from_text(text_chunks, chapter_number, topic_name):
    context = "\n\n".join(text_chunks)
    prompt = build_extraction_prompt(context)

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ----------- Utility: Extract Text from PDF ------------
def extract_text_from_pdf(pdf_path, start_page=0, end_page=None):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        end_page = end_page or len(pdf.pages)
        for i in range(start_page, end_page):
            text = pdf.pages[i].extract_text()
            if text:
                all_text += f"\n\n{text}"
    return all_text

# ----------- Optional OCR -------------------------------
def extract_text_via_ocr(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# ----------- Prompt Template ----------------------------
def build_extraction_prompt(context):
    return f"""You are an expert assistant specialized in Mathematics and LaTeX formatting.
Your task is to extract all practice questions and question-like illustrations from the following text snippet from the RD Sharma textbook. [cite: 13, 25]

Follow these instructions carefully:
1.  **Extract Questions Only**: Identify and extract only the questions. These can be from practice exercises or illustrations that pose a question. [cite: 15, 25]
2.  **Ignore Non-Question Content**: You MUST ignore all other text, including theory, introductory paragraphs, solved examples, hints, and solutions. [cite: 24]
3.  **Preserve Mathematical Formatting**: Convert all mathematical expressions, symbols, equations, fractions, matrices, and notations accurately into LaTeX format. [cite: 26, 27, 28, 29]
4.  **Output Format**: Return the questions as a Python list of strings, where each string is a single, complete, LaTeX-formatted question. [cite: 30]

EXAMPLE:
If the text contains "Question 1: If P(A) = 0.5, find P(A').", your output should be a list containing the string:
"If $P(A) = 0.5$, find $P(A')$."

Now, process the following text:

--- TEXT SNIPPET ---
{context}
--- END TEXT SNIPPET ---

Return ONLY a Python list of LaTeX-formatted question strings."""
#     return f""""Extract **every question or example problem** from the following text as LaTeX-formatted items. Do not summarize. Maintain mathematical notation where appropriate."

# Extract ONLY the questions and examples(ignore solutions and theory). Output the result in LaTeX format. Do not hallucinate content.


# Preserve mathematical symbols and formatting (fractions, roots, summation, etc.)

# Text:
# ----
# {context}
# ----

# LaTeX Output:
# \\begin{{enumerate}}
# \\item ...
# \\end{{enumerate}}"""


# ----------- Chunking & Vector DB ------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

def build_vector_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    temp_dir = tempfile.mkdtemp()
    try:
        db = FAISS.from_texts(chunks, embeddings)
        db.save_local(temp_dir)
        return FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
    finally:
        shutil.rmtree(temp_dir)

# ----------- Main Function -------------------------------
def extract_questions_from_chapter(pdf_path, chapter_number, topic_name):
    # 1. Extract text
    text = extract_text_from_pdf(pdf_path)

    # 2. Chunking
    chunks = chunk_text(text)
    print(f"✅ Total text chunks extracted from PDF: {len(chunks)}")

    matched_chunks = [chunk for chunk in chunks if topic_name.lower() in chunk.lower()]
    print(f"🔍 Chunks matching the topic '{topic_name}': {len(matched_chunks)}")

    vector_db = build_vector_index(chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    relevant_docs = retriever.get_relevant_documents(f"Chapter {chapter_number} {topic_name}")

    # 3. Call Groq API through helper function
    result = generate_latex_questions_from_text([doc.page_content for doc in relevant_docs], chapter_number, topic_name)

    # 4. Output
    os.makedirs("output", exist_ok=True)
    output_path = f"output/chapter_{chapter_number}_{topic_name.replace(' ', '_')}.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\n✅ Extracted questions saved to: {output_path}")
    return result


# ----------- CLI Interface -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RD Sharma Question Extractor (Groq-based)")
    parser.add_argument("--chapter", type=str, required=True, help="Chapter number (e.g., 30)")
    parser.add_argument("--topic", type=str, required=True, help="Topic name (e.g., 30.3 Conditional Probability)")
    parser.add_argument("--pdf", type=str, default="data\\RD-SHARMA CLASS 12TH VOLUME 2 MCQS (R.D.SHARMA) (Z-Library).pdf", help="Path to RD Sharma PDF")
    args = parser.parse_args()

    extract_questions_from_chapter(args.pdf, args.chapter, args.topic)
