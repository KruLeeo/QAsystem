# Document QA System

Система вопросов и ответов по документам с поддержкой русского языка.

## Установка

```bash
python -m pip install -r requirements.txt
python scripts/build_index.py
python scripts/ask.py --q "Ваш вопрос?" --provider ollama  --k 5