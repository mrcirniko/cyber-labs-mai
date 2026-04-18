# Лабораторная работа 2: NLP, Цирулев Николай М8О-408Б-22


## Описание

Pipeline локального LLM-инференса через `Ollama`. Используется модель `qwen2.5:0.5b`; скрипт [infer.py](/lab2/infer.py) отправляет 10 запросов из [prompts.json](/lab2/prompts.json) и формирует отчет.

## Запуск

```bash
sudo snap install ollama
ollama pull qwen2.5:0.5b
ollama serve
```
```bash
cd /home/hacker/prog/cyberphis_labs/lab2-nlp-ollama
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python infer.py
```

## Отчет

Готовый отчет: [reports/inference_report.md](/lab2/reports/inference_report.md)

## Вывод

Инференс через локальный Ollama API работает: модель ответила на все 10 запросов, результаты сохранены в Markdown и JSON. Качество ответов у `qwen2.5:0.5b` нестабильное: простые задачи выполняются, но в фактических и сложных вопросах встречаются ошибки и галлюцинации.
