import PyPDF2
import pandas as pd
from typing import List, Dict, Any
import re

class PDFParser:
    def __init__(self):
        self.text_cleanup_patterns = [
            (r'\n+', '\n'),
            (r' +', ' '),
            (r'-\n', ''),
        ]
    
    def parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Парсит PDF и возвращает текст с метаданными по страницам"""
        pages = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Очистка текста
                cleaned_text = self._clean_text(text)
                if cleaned_text.strip():
                    pages.append({
                        'page_number': page_num + 1,
                        'text': cleaned_text,
                        'source_file': file_path.split('/')[-1],
                        'char_count': len(cleaned_text),
                        'word_count': len(cleaned_text.split())
                    })
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста от артефактов"""
        for pattern, replacement in self.text_cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        return text.strip()