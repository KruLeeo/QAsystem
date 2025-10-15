#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.answering import QAEngine
from src.embedding_index import EmbeddingIndex
from src.retrieval import EnsembleRetriever, BM25Retriever
from src.llm_providers import OllamaProvider, LocalLLMAPIProvider, LlamaCPPProvider
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chunks_for_bm25():
    """Загружает чанки для BM25"""
    try:
        with open('./artifacts/chunks.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Chunks file not found, BM25 will be disabled")
        return []

def get_llm_provider(provider_type: str, model_name: str = None) -> object:
    """Возвращает инициализированный LLM провайдер"""
    
    if provider_type == "ollama":
        model = model_name or "deepseek-r1:8b"  # ← исправлено на deepseek
        return OllamaProvider(model_name=model)
    
    elif provider_type == "local-api":
        return LocalLLMAPIProvider()
    
    elif provider_type == "llama-cpp":
        if not model_name:
            raise ValueError("Model path required for llama-cpp provider")
        return LlamaCPPProvider(model_path=model_name)
    
    elif provider_type == "openai":
        from langchain.llms import OpenAI
        return OpenAI(temperature=0.1)
    
    else:
        raise ValueError(f"Unsupported provider: {provider_type}")

def main():
    parser = argparse.ArgumentParser(description='QA System with Llama')
    parser.add_argument('--q', '--question', required=True, help='Question to ask')
    parser.add_argument('--k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--lang', default='ru', help='Language')
    parser.add_argument('--provider', default='ollama', 
                       choices=['ollama', 'local-api', 'llama-cpp', 'openai'],
                       help='LLM provider')
    parser.add_argument('--model', help='Model name/path (for ollama/llama-cpp)')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid retrieval')
    
    args = parser.parse_args()
    
    # Инициализация компонентов
    logger.info("Initializing components...")
    embedding_index = EmbeddingIndex()
    vector_retriever = embedding_index.get_retriever()
    
    # Инициализация retriever
    if args.hybrid:
        chunks = load_chunks_for_bm25()
        if chunks:
            bm25_retriever = BM25Retriever(chunks)
            retriever = EnsembleRetriever(vector_retriever, bm25_retriever)
            logger.info("Using hybrid retrieval")
        else:
            retriever = vector_retriever
            logger.warning("BM25 chunks not found, using vector retrieval only")
    else:
        retriever = vector_retriever
        logger.info("Using vector retrieval")
    
    # Инициализация LLM провайдера
    try:
        llm_provider = get_llm_provider(args.provider, args.model)
        logger.info(f"LLM provider initialized: {args.provider}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM provider: {e}")
        sys.exit(1)
    
    # Создаем QA engine
    qa_engine = QAEngine(retriever, llm_provider)
    
    # Задаем вопрос
    logger.info(f"Processing question: {args.q}")
    result = qa_engine.ask(args.q, args.k, args.lang)
    
    # Выводим результат
    print("\n" + "="*60)
    print(f"Вопрос: {result['question']}")
    print("="*60)
    print(f"Ответ: {result['answer']}")
    print("="*60)
    
    if result['sources']:
        print("\n📚 Источники:")
        for source in result['sources']:
            print(f"   📄 {source['doc_id']} (страницы {source['pages']})")
    
    if result['no_answer']:
        print("\n⚠️  Примечание: ответ помечен как 'no-answer'")
    
    # ИСПРАВЛЕННЫЙ БЛОК - правильные ключи timing
    print(f"\n⏱️  Время обработки: {result['timing']['total_ms']:.2f}ms")
    print(f"   - Retrieval: {result['timing']['retrieval_ms']:.2f}ms")
    print(f"   - LLM: {result['timing']['llm_ms']:.2f}ms")  # ← исправлено здесь

if __name__ == "__main__":
    main()