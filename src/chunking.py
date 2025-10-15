from typing import List, Dict, Any
import re

class ContentAwareChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Разбивает документ на чанки с учетом структуры содержания"""
        chunks = []
        
        for page in pages:
            page_chunks = self._chunk_page_content(page)
            chunks.extend(page_chunks)
        
        return chunks
    
    def _chunk_page_content(self, page: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Разбивает страницу на семантические чанки"""
        text = page['text']
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence
                current_sentences.append(sentence)
            else:
                if current_chunk.strip():
                    chunks.append(self._create_chunk_metadata(
                        current_chunk.strip(), 
                        current_sentences,
                        page
                    ))
                
                # Добавляем overlap
                overlap_sentences = current_sentences[-self._get_overlap_sentence_count():]
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_sentences = overlap_sentences + [sentence]
        
        # Добавляем последний чанк
        if current_chunk.strip():
            chunks.append(self._create_chunk_metadata(
                current_chunk.strip(),
                current_sentences,
                page
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        sentence_endings = r'[.!?]+[\s\n]'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentence_count(self) -> int:
        """Вычисляет количество предложений для overlap"""
        return 2  # Эмпирическое значение
    
    def _create_chunk_metadata(self, chunk_text: str, sentences: List[str], 
                             page: Dict[str, Any]) -> Dict[str, Any]:
        """Создает метаданные для чанка"""
        return {
            'text': chunk_text,
            'source_file': page['source_file'],
            'page_number': page['page_number'],
            'chunk_id': f"{page['source_file']}_p{page['page_number']}_c{len(chunk_text)}",
            'char_count': len(chunk_text),
            'word_count': len(chunk_text.split()),
            'sentence_count': len(sentences)
        }