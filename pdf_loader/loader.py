import os
import fitz  # PyMuPDF
from utils import logger


def load_pdf_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        logger.error("PDFLoader", f"File not found: {pdf_path}")
        return ""

    if not pdf_path.lower().endswith(".pdf"):
        logger.error("PDFLoader", f"Not a PDF: {pdf_path}")
        return ""

    try:
        doc        = fitz.open(pdf_path)
        pages_text = []

        for i in range(len(doc)):
            text = doc.load_page(i).get_text("text")
            if text.strip():
                pages_text.append(text.strip())

        doc.close()
        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            logger.warning("PDFLoader", "No extractable text (scanned PDF?).")
            return ""

        logger.success("PDFLoader", f"Extracted {len(full_text):,} chars from {len(pages_text)} pages.")
        return full_text

    except Exception as e:
        logger.error("PDFLoader", f"Error reading PDF: {e}")
        return ""