import pdfplumber
from docx import Document
import os

def convert_pdf_to_docx(pdf_path: str, docx_path: str):
    document = Document()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    document.add_paragraph(line)

    document.save(docx_path)
