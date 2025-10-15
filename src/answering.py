from langchain.schema import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StrictContextAnswerParser(BaseOutputParser):
    """Парсер для строгого ответа по контексту"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Парсит ответ LLM на содержание и источники"""
        lines = text.strip().split('\n')
        answer_lines = []
        source_lines = []
        in_sources = False
        
        for line in lines:
            if line.lower().startswith('источники:'):
                in_sources = True
                continue
            if in_sources:
                source_lines.append(line.strip())
            else:
                answer_lines.append(line.strip())
        
        answer = '\n'.join(answer_lines).strip()
        sources = [s for s in source_lines if s]
        
        return {
            'answer': answer,
            'sources': sources,
            'no_answer': self._is_no_answer(answer)
        }
    
    def _is_no_answer(self, answer: str) -> bool:
        """Определяет, является ли ответ отказом"""
        no_answer_indicators = [
            'не могу ответить',
            'информация отсутствует',
            'в предоставленном контексте нет',
            'нет информации',
            'не найдено в контексте'
        ]
        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in no_answer_indicators)

class QAEngine:
    def __init__(self, retriever, llm_provider):
        self.retriever = retriever
        self.llm_provider = llm_provider
        
        self.prompt_template = PromptTemplate(
            template="""Ты - ассистент, который отвечает ТОЛЬКО на основе предоставленного контекста. 

Контекст:
{context}

Вопрос: {question}

Инструкции:
1. Ответь на вопрос, используя ТОЛЬКО информацию из контекста
2. Если в контексте недостаточно информации для ответа, вежливо откажись отвечать
3. В конце ответа добавь раздел "Источники:" с перечислением использованных источников в формате: doc_id (страницы X-Y)

Ответ:""",
            input_variables=["context", "question"]
        )
        
        self.parser = StrictContextAnswerParser()
    
    def ask(self, question: str, k: int = 5, lang: str = 'ru') -> Dict[str, Any]:
        """Основной метод для вопросов"""
        start_time = time.time()
        
        # Retrieval
        retrieved_chunks = self.retriever.search(question, k)
        retrieval_time = time.time() - start_time
        
        # Подготавливаем контекст
        context = self._format_context(retrieved_chunks)
        
        # Генерируем ответ
        llm_start = time.time()
        prompt = self.prompt_template.format(context=context, question=question)
        raw_response = self.llm_provider.generate(prompt)
        llm_time = time.time() - llm_start
        
        # Парсим ответ
        parsed_response = self.parser.parse(raw_response)
        
        # Форматируем источники
        formatted_sources = self._extract_source_info(retrieved_chunks)
        
        total_time = time.time() - start_time
        
        # Логируем
        self._log_query(
            question=question,
            retrieved_chunks=retrieved_chunks,
            response=parsed_response,
            retrieval_time=retrieval_time,
            llm_time=llm_time,
            total_time=total_time
        )
        
        return {
    'question': question,
    'answer': parsed_response['answer'],
    'sources': formatted_sources,
    'retrieved_chunks': retrieved_chunks,
    'no_answer': parsed_response['no_answer'],
    'timing': {
        'retrieval_ms': retrieval_time * 1000,
        'llm_ms': llm_time * 1000,        # ← правильный ключ
        'total_ms': total_time * 1000
    },
    'retrieved_ids': [chunk['chunk_id'] for chunk in retrieved_chunks]
}
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Форматирует контекст для LLM"""
        context_parts = []
        for chunk in chunks:
            source_info = f"[Источник: {chunk['source_file']}, страница {chunk['page_number']}]"
            context_parts.append(f"{source_info}\n{chunk['text']}\n")
        return "\n".join(context_parts)
    
    def _extract_source_info(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Извлекает информацию об источниках"""
        source_map = {}
        
        for chunk in chunks:
            source_key = chunk['source_file']
            if source_key not in source_map:
                source_map[source_key] = {
                    'doc_id': source_key,
                    'pages': set(),
                    'chunk_ids': []
                }
            
            source_map[source_key]['pages'].add(chunk['page_number'])
            source_map[source_key]['chunk_ids'].append(chunk['chunk_id'])
        
        # Форматируем диапазоны страниц
        formatted_sources = []
        for source in source_map.values():
            pages = sorted(source['pages'])
            page_ranges = self._format_page_ranges(pages)
            formatted_sources.append({
                'doc_id': source['doc_id'],
                'pages': page_ranges,
                'page_list': pages,
                'chunk_count': len(source['chunk_ids'])
            })
        
        return formatted_sources
    
    def _format_page_ranges(self, pages: List[int]) -> str:
        """Форматирует список страниц в диапазоны"""
        if not pages:
            return ""
        
        ranges = []
        start = pages[0]
        end = pages[0]
        
        for page in pages[1:]:
            if page == end + 1:
                end = page
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = page
                end = page
        
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ", ".join(ranges)
    
    def _log_query(self, **kwargs):
        """Логирует информацию о запросе"""
        log_data = {
            'question': kwargs['question'],
            'retrieved_ids': kwargs['retrieved_chunks'],
            'retrieval_time_ms': kwargs['retrieval_time'] * 1000,
            'llm_time_ms': kwargs['llm_time'] * 1000,
            'total_time_ms': kwargs['total_time'] * 1000,
            'no_answer': kwargs['response']['no_answer']
        }
        logger.info(f"Query processed: {log_data}")