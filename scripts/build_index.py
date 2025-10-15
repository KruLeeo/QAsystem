#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.file_parser import UniversalParser  # ← изменили импорт
from src.chunking import ContentAwareChunker
from src.embedding_index import EmbeddingIndex
import glob
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Конфигурация - теперь поддерживаем разные форматы
    file_directory = "./data/raw/*.*"  # ← изменили маску
    persist_directory = "./artifacts/chroma_db"
    
    # Инициализация компонентов
    parser = UniversalParser()  # ← используем универсальный парсер
    chunker = ContentAwareChunker()
    index_builder = EmbeddingIndex(persist_directory)
    
    # Парсинг файлов разных форматов
    all_pages = []
    supported_formats = ['*.pdf', '*.txt', '*.docx', '*.doc']
    
    for file_format in supported_formats:
        file_pattern = f"./data/raw/{file_format}"
        files = glob.glob(file_pattern)
        
        for file_path in files:
            logger.info(f"Processing {file_path}")
            try:
                pages = parser.parse_file(file_path)
                all_pages.extend(pages)
                logger.info(f"Extracted {len(pages)} pages from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    # Чанкинг
    logger.info("Chunking documents...")
    chunks = chunker.chunk_document(all_pages)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Построение индекса
    logger.info("Building embedding index...")
    stats = index_builder.build_index(chunks)
    
    # Сохранение статистики
    with open("./artifacts/index_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Index built successfully!")
    logger.info(f"Statistics: {stats}")

if __name__ == "__main__":
    main()