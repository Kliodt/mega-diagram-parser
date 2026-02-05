# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости для OpenCV и Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY core/ ./core/
COPY model/ ./model/


# Устанавливаем переменную окружения для tesseract
ENV TESSERACT_CMD=/usr/bin/tesseract

# Устанавливаем PYTHONPATH чтобы импорты работали
ENV PYTHONPATH=/app/core

# Открываем порт для FastAPI
EXPOSE 8000

# Команда запуска сервера
CMD ["uvicorn", "core.controller:app", "--host", "0.0.0.0", "--port", "8000"]
