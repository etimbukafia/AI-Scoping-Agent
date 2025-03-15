import textract
import PyPDF2
from pathlib import Path
import os
import logging
import re
import io

def extract_text_from_pdf(pdf_source):
    """
    Extracts all text from the PDF using PyPDF2.
    
    :param pdf_source: Either a path to the PDF file or PDF binary data.
    :return: The full text content of the PDF.
    """
    text = ""
    try:      
        # Handle both file paths and binary data
        if isinstance(pdf_source, bytes):
            # It's binary data
          
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_source))
        else:
            # It's a file path
            with open(pdf_source, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                
        # Extract text from all pages
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                logging.warning("No text found on page %s.", page_num)
    except Exception as e:
        logging.exception("Failed to extract text from PDF: %s", e)
    
    return text

def extract_from_txt(txt_source):
    """
    Extracts text from a txt file or binary data.
    
    :param txt_source: Either a path to a txt file or binary/string data.
    :return: The text content.
    """
    # If it's a file path
    if isinstance(txt_source, str) and os.path.exists(txt_source):
        with open(txt_source, "r") as f:
            text = f.read()
        return text
        
    # If it's binary data
    if isinstance(txt_source, bytes):
        # Try different encodings
        encodings = ["utf-8", "latin-1", "windows-1252", "ascii", "utf-16"]
        for encoding in encodings:
            try:
                return txt_source.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # If all decodings fail
        raise ValueError("Could not decode text file - unknown encoding")
        
    # If it's already a string
    if isinstance(txt_source, str):
        return txt_source
        
    raise ValueError("Unsupported input type for text extraction")

def extract_text_from_doc(file):
    """
    Extracts text from the given document input.
    
    - If file is bytes, determine type and process accordingly.
    - If file is a string that is not a valid file path, assume it's already the content.
    - If file is a valid file path, process it based on its extension.
    """
    # Handle binary data
    if isinstance(file, bytes):
        # Check if it's a PDF by looking at the magic bytes
        if file.startswith(b'%PDF-'):
            return extract_text_from_pdf(file)
        else:
            # Assume it's a text file and try to decode it
            return extract_from_txt(file)
    
    # Handle string (either path or content)
    if isinstance(file, str):
        # Check if it's a valid file path
        if os.path.exists(file):
            # Process based on file extension
            if file.endswith((".doc", ".docx")):
                return textract.process(file).decode("utf-8")
            elif file.endswith(".pdf"):
                return extract_text_from_pdf(file)
            elif file.endswith(".txt"):
                return extract_from_txt(file)
            else:
                raise ValueError("Unsupported file type")
        else:
            # Assume it's already the content
            return file

    raise ValueError("Unsupported input type for extract_text_from_doc")
    
def extract_introduction(text):
    """
    Extracts text from the 'Introduction' section.
    Assumes the section starts with a line that exactly reads '2 Introduction'
    and ends right before the next section heading (i.e. a line starting with a digit and a space).
    """
    pattern = r"^2 Introduction\s*\n([\s\S]*?)(?=^\d+ )"
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    else:
        return "Introduction section not found."
    

