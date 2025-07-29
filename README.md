# Workable.

# 📘 LaTeX Question Generator using RAG and Groq LLaMA 3

## 🔍 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to automatically generate LaTeX-formatted questions from textbook content. The goal is to extract meaningful question-answer pairs from chapter-wise topic content, suitable for educational applications like quizzes, worksheets, or exam preparation.

Users input a chapter number and topic name, and the system:
1. Retrieves relevant `.txt` content files.
2. Chunks and aggregates the text.
3. Prompts a powerful LLM (LLaMA 3 via Groq) to generate LaTeX-style questions.
4. Outputs a structured LaTeX block containing the generated content.

---

## 🧰 Tools & Technologies Used

- **Python**: Core programming language.
- **Groq API**: For high-speed inference using `llama3-70b-8192`.
- **LLaMA 3**: Large Language Model used for extracting and generating LaTeX Q&A pairs.
- **File I/O**: To manage topic-wise `.txt` input files.
- **Notebook/CLI Interface**: Users can interact via a demo notebook or `extractor.py` terminal script.

---

## 🧠 Prompting Strategy

We use a structured prompt that provides the entire topic context (joined from multiple `.txt` chunks) and asks the model to:
- Extract meaningful questions.
- Format everything in proper LaTeX (e.g., `\begin{question}`, `\question`, `\end{question}`).

Example system prompt:
```text
You are an AI assistant extracting questions from a Class 12 Math textbook.

Extract ONLY the questions (ignore solutions and theory). Output the result in LaTeX format.

Preserve mathematical symbols and formatting (fractions, roots, summation, etc.)

Text:
----
{context}
----

LaTeX Output:
\\begin{{enumerate}}
\\item ...
\\end{{enumerate}}"""

#❗ Challenges Faced
1. Encoding Errors
Problem: Some .txt files had non-UTF-8 characters, leading to decode errors.

Solution: Wrapped file reads in a try-except block and added support for errors='ignore' or switched encoding to ISO-8859-1 when necessary.

#How to Run
Option 1: Terminal
python extractor.py --chapter 19 --topic "19.4 Properties of Definite Integrals"