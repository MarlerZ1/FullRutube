from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView

# import ai libs
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Загрузка модели и токенизатора
model = AutoModelForSequenceClassification.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

# Используем устройство CUDA, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Django rest framework class-based view с произвольным возвращаемым  json значением
class AIAnalysis(APIView):
    # Метод гет, наследуемымый от APIView. Вызывается при отправке гет-запроса на адрес из urls.py, по которому подключена данная class-based view.
    def get(self, request):
        answer = {}

        # Ввод заголовка и описания видео
        title = request.data['video_title']
        description = request.data['video_description']

        # Объединение и нормализация текста
        input_text = title + " " + description
        normalized_text = self.normalize_text(input_text)

        # Токенизация
        encoding = tokenizer(normalized_text, return_tensors='pt', truncation=True, padding=True, max_length=256)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Предсказание
        with torch.no_grad():
            outputs = model(**encoding)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Загрузка кодировщика меток, использованного для обучения
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load('./saved_model/label_encoder_classes.npy', allow_pickle=True)

        # Вывод тегов с вероятностями
        top_k = 5  # Количество выводимых тегов
        probabilities, indices = torch.topk(probabilities, top_k, dim=1)

        print("Предсказанные теги с вероятностями:")
        for prob, idx in zip(probabilities[0], indices[0]):
            tag = label_encoder.inverse_transform([idx.item()])[0]
            answer[tag] = prob.item()
            print(f"{tag}: {prob.item():.4f}")

        return Response(answer)

    # Функция нормализации текста
    def normalize_text(self, text):
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
        text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации и неалфавитных символов
        return text
