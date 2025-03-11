# test_pdf_extraction_fitz.py
import sys
from OllamaTest1 import extract_text_from_pdf  # Ensure this file is in your PYTHONPATH

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_extraction_fitz.py </Users/aotalora/Downloads/MyModels/PDFs/McC.pdf>")
        return

    pdf_path = sys.argv[1]
    text = extract_text_from_pdf(pdf_path)

    if text:
        print("Extracted text using PyMuPDF:")
        print(text)
    else:
        print("No text was extracted from the PDF.")

if __name__ == '__main__':
    main()