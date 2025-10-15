from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Скачаем необходимые данные nltk при первом запуске
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(chunk['text']) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        return word_tokenize(text.lower())
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Поиск с использованием BM25"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Сортируем результаты по релевантности
        scored_chunks = list(zip(scores, self.chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, chunk in scored_chunks[:k]:
            results.append({
                **chunk,
                'score': score,
                'retriever': 'bm25'
            })
        
        return results

class EnsembleRetriever:
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Гибридный поиск с RRF (Reciprocal Rank Fusion)"""
        vector_results = self.vector_retriever.search(query, k * 2)
        bm25_results = self.bm25_retriever.search(query, k * 2)
        
        # RRF scoring
        rrf_scores = {}
        
        for rank, result in enumerate(vector_results):
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank + 1)
        
        for rank, result in enumerate(bm25_results):
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank + 1)
        
        # Объединяем результаты
        all_results = {r['chunk_id']: r for r in vector_results + bm25_results}
        
        # Сортируем по RRF score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: rrf_scores.get(x['chunk_id'], 0),
            reverse=True
        )
        
        # Добавляем итоговый score
        for result in sorted_results[:k]:
            result['ensemble_score'] = rrf_scores.get(result['chunk_id'], 0)
            result['retriever'] = 'ensemble'
        
        return sorted_results[:k]