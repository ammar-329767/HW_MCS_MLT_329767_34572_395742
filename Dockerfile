FROM python:3.10-slim

WORKDIR /app

# تثبيت المتطلبات أولاً
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كل الملفات
COPY . .

# تدريب النموذج وحفظه
RUN python train_model.py

# التحقق من وجود النموذج
RUN ls -la *.keras *.h5 2>/dev/null || echo "No model files found"

EXPOSE 5000

CMD ["python", "app.py"]
