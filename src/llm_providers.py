from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Optional, Dict, Any
import requests
import json
import logging
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

logger = logging.getLogger(__name__)

class OllamaProvider:
    """
    Провайдер для работы с Ollama локальными моделями
    """
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama2:7b')
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Инициализирует Ollama LLM"""
        try:
            # Проверяем доступность Ollama
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not available at {self.base_url}")
            
            # Получаем список доступных моделей
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                logger.info(f"Trying to pull model {self.model_name}...")
                # Пытаемся скачать модель
                pull_response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={'name': self.model_name},
                    timeout=300
                )
                if pull_response.status_code != 200:
                    raise ValueError(f"Failed to pull model {self.model_name}")
            
            llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature=0.1,  # Низкая температура для более детерминированных ответов
                top_p=0.9,
                num_predict=2048,  # Максимальная длина ответа
                num_ctx=4096,  # Размер контекста
            )
            logger.info(f"Ollama provider initialized with model: {self.model_name}")
            return llm
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {self.base_url}")
            logger.info("Please make sure Ollama is running: 'ollama serve'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ на промпт"""
        try:
            # Добавляем системный промпт для русскоязычных ответов
            enhanced_prompt = self._enhance_prompt(prompt)
            
            response = self.llm.invoke(enhanced_prompt, **kwargs)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            return "Извините, произошла ошибка при генерации ответа. Пожалуйста, проверьте подключение к локальной модели."
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Улучшает промпт для лучших русскоязычных ответов"""
        system_instruction = """Ты - полезный ассистент, который отвечает на вопросы на РУССКОМ языке. 
Отвечай четко и по делу, используя только предоставленную информацию. 
Если информации недостаточно, вежливо откажись отвечать.
В конце ответа обязательно укажи источники информации в формате: "Источники: название_файла (страницы X-Y)"."""
        
        return f"{system_instruction}\n\n{prompt}"


class LlamaCPPProvider:
    """
    Альтернативный провайдер через llama-cpp-python
    """
    
    def __init__(self, model_path: str = None, n_ctx: int = 4096):
        self.model_path = model_path or os.getenv('LLAMA_CPP_MODEL_PATH')
        if not self.model_path:
            raise ValueError("Model path must be provided for LlamaCPPProvider")
        
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=8,
                verbose=False,
                use_mlock=True,
                use_mmap=True,
            )
            logger.info(f"LlamaCPP provider initialized with model: {self.model_path}")
        except ImportError:
            raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaCPP: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ через llama.cpp"""
        try:
            # Улучшаем промпт
            enhanced_prompt = self._enhance_prompt(prompt)
            
            response = self.llm(
                enhanced_prompt,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                echo=False,
                stop=["\n\n", "Источники:", "Sources:"],
                stream=False
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error generating response with LlamaCPP: {e}")
            return "Извините, произошла ошибка при генерации ответа."
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Улучшает промпт для лучших ответов"""
        system_instruction = "Ты - полезный ассистент. Отвечай на РУССКОМ языке четко и по делу. Используй только предоставленную информацию."
        return f"{system_instruction}\n\n{prompt}"


class LocalLLMAPIProvider:
    """
    Провайдер для локального LLM API (например, text-generation-webui, Oobabooga)
    """
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('LOCAL_API_URL', 'http://localhost:5000')
        self.generate_url = f"{self.base_url}/api/v1/generate"
        self._test_connection()
    
    def _test_connection(self):
        """Проверяет подключение к локальному API"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/model", timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                logger.info(f"Connected to local API. Model: {model_info.get('model_name', 'Unknown')}")
            else:
                logger.warning(f"Local API returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to local API: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ через локальное API"""
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
            "typical_p": 0.95,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "seed": -1,
            "truncation_length": 4096,
            "stop": ["\n\n", "Источники:", "Sources:"]
        }
        
        try:
            response = requests.post(self.generate_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result['results'][0]['text'].strip()
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to local API at {self.base_url}")
            return "Ошибка: не удалось подключиться к локальной модели. Убедитесь, что API сервер запущен."
        except Exception as e:
            logger.error(f"Local API request failed: {e}")
            return "Ошибка при обращении к локальной модели."


class OpenAIProvider:
    """
    Провайдер для OpenAI API (резервный вариант)
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        try:
            from langchain.llms import OpenAI
            from langchain.chat_models import ChatOpenAI
            
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            
            os.environ['OPENAI_API_KEY'] = self.api_key
            
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.1,
                max_tokens=1024,
            )
            logger.info(f"OpenAI provider initialized with model: {model}")
            
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Генерирует ответ через OpenAI API"""
        try:
            from langchain.schema import HumanMessage
            
            messages = [
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return "Извините, произошла ошибка при обращении к OpenAI API."


class FallbackProvider:
    """
    Провайдер с fallback механизмом
    """
    
    def __init__(self, providers: list):
        self.providers = providers
        self.current_provider_idx = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Пытается сгенерировать ответ через доступные провайдеры"""
        for i in range(len(self.providers)):
            provider = self.providers[(self.current_provider_idx + i) % len(self.providers)]
            try:
                logger.info(f"Trying provider: {type(provider).__name__}")
                response = provider.generate(prompt, **kwargs)
                if response and not response.startswith("Ошибка"):
                    self.current_provider_idx = (self.current_provider_idx + i) % len(self.providers)
                    return response
            except Exception as e:
                logger.warning(f"Provider {type(provider).__name__} failed: {e}")
                continue
        
        return "Извините, все провайдеры недоступны. Пожалуйста, проверьте настройки."


def get_llm_provider(provider_type: str = None, **kwargs) -> object:
    """
    Фабрика для создания LLM провайдеров
    
    Args:
        provider_type: Тип провайдера ('ollama', 'llama-cpp', 'local-api', 'openai')
        **kwargs: Дополнительные параметры для провайдера
    
    Returns:
        Инициализированный провайдер
    """
    provider_type = provider_type or os.getenv('LLM_PROVIDER', 'ollama')
    
    try:
        if provider_type == "ollama":
            return OllamaProvider(
                model_name=kwargs.get('model_name'),
                base_url=kwargs.get('base_url')
            )
        
        elif provider_type == "llama-cpp":
            return LlamaCPPProvider(
                model_path=kwargs.get('model_path'),
                n_ctx=kwargs.get('n_ctx', 4096)
            )
        
        elif provider_type == "local-api":
            return LocalLLMAPIProvider(
                base_url=kwargs.get('base_url')
            )
        
        elif provider_type == "openai":
            return OpenAIProvider(
                api_key=kwargs.get('api_key'),
                model=kwargs.get('model', 'gpt-3.5-turbo')
            )
        
        elif provider_type == "fallback":
            # Создаем цепочку fallback провайдеров
            providers = []
            if kwargs.get('ollama_enabled', True):
                try:
                    providers.append(OllamaProvider())
                except:
                    pass
            
            if kwargs.get('local_api_enabled', True):
                try:
                    providers.append(LocalLLMAPIProvider())
                except:
                    pass
            
            if kwargs.get('openai_enabled', False):
                try:
                    providers.append(OpenAIProvider())
                except:
                    pass
            
            if not providers:
                raise ValueError("No available providers for fallback mode")
            
            return FallbackProvider(providers)
        
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    except Exception as e:
        logger.error(f"Failed to create provider {provider_type}: {e}")
        raise


# Утилиты для работы с моделями
def list_available_ollama_models(base_url: str = None) -> list:
    """Возвращает список доступных моделей в Ollama"""
    base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except:
        return []


def download_ollama_model(model_name: str, base_url: str = None) -> bool:
    """Скачивает модель через Ollama API"""
    base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    try:
        response = requests.post(
            f"{base_url}/api/pull",
            json={'name': model_name},
            timeout=300,
            stream=True
        )
        
        if response.status_code == 200:
            logger.info(f"Started downloading model: {model_name}")
            # Можно добавить прогресс-бар здесь
            for line in response.iter_lines():
                if line:
                    try:
                        progress = json.loads(line)
                        if 'status' in progress:
                            logger.info(f"Download status: {progress['status']}")
                    except:
                        pass
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        return False