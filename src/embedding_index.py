import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmbeddingIndex:
    def __init__(self, persist_directory: str = "./artifacts/chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Создаем клиент Chroma с правильными настройками
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def build_index(self, chunks: List[Dict[str, Any]], collection_name: str = "documents") -> Dict[str, Any]:
        """Строит индекс и возвращает статистику"""
        start_time = time.time()
        
        # Создаем или получаем коллекцию
        try:
            collection = self.client.get_collection(collection_name)
            self.client.delete_collection(collection_name)  # Удаляем старую
        except:
            pass  # Коллекции не существует
        
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Document chunks with metadata"}
        )
        
        # Подготавливаем данные
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['text'])
            metadatas.append({
                'source_file': chunk['source_file'],
                'page_number': chunk['page_number'],
                'chunk_id': chunk['chunk_id'],
                'char_count': chunk['char_count'],
                'word_count': chunk['word_count']
            })
            ids.append(chunk['chunk_id'])
        
        # Добавляем в коллекцию
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # В новой версии Chroma persist() не нужен - данные сохраняются автоматически
        end_time = time.time()
        build_time = end_time - start_time
        
        # Статистика
        stats = {
            'chunk_count': len(chunks),
            'build_time_seconds': build_time,
            'index_size_mb': self._get_index_size(),
            'collection_name': collection_name
        }
        
        logger.info(f"Index built: {stats}")
        return stats
    
    def _get_index_size(self) -> float:
        """Вычисляет размер индекса в MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.persist_directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return round(total_size / (1024 * 1024), 2)
    
    def get_retriever(self, collection_name: str = "documents"):
        """Возвращает retriever для поиска"""
        collection = self.client.get_collection(collection_name)
        return VectorRetriever(collection, self.embedding_model)

class VectorRetriever:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Поиск похожих чанков"""
        # Генерируем эмбеддинг для запроса
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Ищем в коллекции
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
        
        # Форматируем результаты
        formatted_results = []
        for i, (metadata, document, distance) in enumerate(zip(
            results['metadatas'][0],
            results['documents'][0],
            results['distances'][0]
        )):
            formatted_results.append({
                'chunk_id': metadata['chunk_id'],
                'text': document,
                'source_file': metadata['source_file'],
                'page_number': metadata['page_number'],
                'score': 1 - distance,  # Конвертируем в схожесть
                'metadata': metadata
            })
        
        return formatted_results