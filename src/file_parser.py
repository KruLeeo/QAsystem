import os
import PyPDF2
from typing import List, Dict, Any
import re
try:
    from docx import Document
except ImportError:
    Document = None

class UniversalParser:
    def __init__(self):
        self.text_cleanup_patterns = [
            (r'\n+', '\n'),
            (r' +', ' '),
            (r'-\n', ''),
        ]
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсит файлы разных форматов"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._parse_pdf(file_path)
        elif file_ext == '.txt':
            return self._parse_txt(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсит PDF файлы"""
        pages = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                cleaned_text = self._clean_text(text)
                if cleaned_text.strip():
                    pages.append({
                        'page_number': page_num + 1,
                        'text': cleaned_text,
                        'source_file': os.path.basename(file_path),
                        'char_count': len(cleaned_text),
                        'word_count': len(cleaned_text.split()),
                        'file_type': 'pdf'
                    })
        
        return pages
    
    def _parse_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсит TXT файлы"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp1251') as file:
                content = file.read()
        
        # Разбиваем текстовый файл на "страницы" по 1000 символов
        chunk_size = 1000
        pages = []
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            cleaned_text = self._clean_text(chunk)
            
            if cleaned_text.strip():
                pages.append({
                    'page_number': (i // chunk_size) + 1,
                    'text': cleaned_text,
                    'source_file': os.path.basename(file_path),
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split()),
                    'file_type': 'txt'
                })
        
        return pages
    
    def _parse_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсит DOCX файлы"""
        if Document is None:
            raise ImportError("python-docx is not installed. Run: pip install python-docx")
        
        doc = Document(file_path)
        pages = []
        
        # Собираем весь текст
        full_text = ""
        for paragraph in doc.paragraphs:
            full_text += paragraph.text + "\n"
        
        # Разбиваем на "страницы" по 1000 символов
        chunk_size = 1000
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            cleaned_text = self._clean_text(chunk)
            
            if cleaned_text.strip():
                pages.append({
                    'page_number': (i // chunk_size) + 1,
                    'text': cleaned_text,
                    'source_file': os.path.basename(file_path),
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split()),
                    'file_type': 'docx'
                })
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста от артефактов"""
        for pattern, replacement in self.text_cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        return text.strip()