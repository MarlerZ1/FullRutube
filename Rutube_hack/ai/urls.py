from django.contrib import admin
from django.urls import path

from ai.views import AIAnalysis
#устанавливаем название приложения для корректной работы urls.py
app_name = "ai"
urlpatterns = [
    # добавляем class based view для обработки адреса **/ai/get_analysis
    path('get_analysis', AIAnalysis.as_view()),
]
