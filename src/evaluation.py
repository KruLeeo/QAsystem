import pandas as pd
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import ndcg_score

class RetrievalEvaluator:
    def __init__(self, golden_set_path: str):
        self.golden_set = pd.read_csv(golden_set_path)
    
    def evaluate_retrieval(self, retrieval_results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Вычисляет метрики retrieval"""
        recalls = []
        mrrs = []
        ndcgs = []
        
        for _, row in self.golden_set.iterrows():
            qid = row['qid']
            expected_doc = row['expected_doc_id']
            expected_pages = set(map(int, str(row['expected_pages']).split(',')))
            
            if qid in retrieval_results:
                results = retrieval_results[qid]
                recall = self._calculate_recall(results, expected_doc, expected_pages)
                mrr = self._calculate_mrr(results, expected_doc, expected_pages)
                ndcg = self._calculate_ndcg(results, expected_doc, expected_pages)
                
                recalls.append(recall)
                mrrs.append(mrr)
                ndcgs.append(ndcg)
        
        return {
            'recall@5': np.mean(recalls),
            'mrr': np.mean(mrrs),
            'ndcg@5': np.mean(ndcgs)
        }
    
    def _calculate_recall(self, results: List[Dict], expected_doc: str, 
                         expected_pages: set) -> float:
        """Recall@5: есть ли релевантный документ в топ-5"""
        for result in results[:5]:
            if (result['source_file'] == expected_doc and 
                result['page_number'] in expected_pages):
                return 1.0
        return 0.0
    
    def _calculate_mrr(self, results: List[Dict], expected_doc: str, 
                      expected_pages: set) -> float:
        """Mean Reciprocal Rank"""
        for rank, result in enumerate(results, 1):
            if (result['source_file'] == expected_doc and 
                result['page_number'] in expected_pages):
                return 1.0 / rank
        return 0.0
    
    def _calculate_ndcg(self, results: List[Dict], expected_doc: str, 
                       expected_pages: set) -> float:
        """nDCG@5"""
        relevance_scores = []
        for result in results[:5]:
            if (result['source_file'] == expected_doc and 
                result['page_number'] in expected_pages):
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.0)
        
        if not relevance_scores:
            return 0.0
        
        # Идеальный порядок
        ideal_scores = sorted(relevance_scores, reverse=True)
        
        return ndcg_score([ideal_scores], [relevance_scores])

class AnswerEvaluator:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
    
    def evaluate_answers(self, answers: List[Dict]) -> Dict[str, float]:
        """Оценивает ответы по нескольким метрикам"""
        faithfulness_scores = []
        source_page_scores = []
        no_answer_rates = []
        trust_scores = []
        
        for answer in answers:
            faithfulness = self._evaluate_faithfulness(answer)
            source_page = self._evaluate_source_page(answer)
            no_answer = 1.0 if answer.get('no_answer', False) else 0.0
            
            trust_score = self._calculate_trust_score(answer, faithfulness, source_page)
            
            faithfulness_scores.append(faithfulness)
            source_page_scores.append(source_page)
            no_answer_rates.append(no_answer)
            trust_scores.append(trust_score)
        
        return {
            'faithfulness': np.mean(faithfulness_scores),
            'source_page': np.mean(source_page_scores),
            'no_answer_rate': np.mean(no_answer_rates),
            'trust_score': np.mean(trust_scores)
        }
    
    def _evaluate_faithfulness(self, answer: Dict) -> float:
        """Оценивает, подтверждены ли утверждения контекстом"""
        # Упрощенная версия - можно использовать LLM для проверки
        context_text = " ".join([chunk['text'] for chunk in answer['retrieved_chunks']])
        answer_text = answer['answer']
        
        # Простая эвристика - проверка ключевых слов в контексте
        answer_words = set(answer_text.lower().split())
        context_words = set(context_text.lower().split())
        
        if len(answer_words) == 0:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words)) / len(answer_words)
        return min(overlap * 2, 1.0)  # Нормализуем к 0-1
    
    def _evaluate_source_page(self, answer: Dict) -> float:
        """Проверяет наличие валидных ссылок на страницы"""
        if not answer.get('sources'):
            return 0.0
        
        valid_sources = 0
        for source in answer['sources']:
            if (source.get('doc_id') and 
                source.get('pages') and 
                source.get('page_list')):
                valid_sources += 1
        
        return valid_sources / len(answer['sources'])
    
    def _calculate_trust_score(self, answer: Dict, faithfulness: float, 
                             source_page: float) -> float:
        """Упрощенный Trust Score"""
        # source_tier_score (A=1, B=0.7, C=0.4)
        source_tier = 1.0  # По умолчанию A
        
        editor_verified = 1.0  # По умолчанию верифицировано
        
        # overlap_with_expected (упрощенно)
        expected_overlap = 1.0 if faithfulness > 0.7 else 0.0
        
        trust_score = (0.5 * source_tier + 
                      0.3 * editor_verified + 
                      0.2 * expected_overlap)
        
        return trust_score