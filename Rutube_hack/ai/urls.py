from django.contrib import admin
from django.urls import path

from ai.views import AIAnalysis

app_name = "ai"
urlpatterns = [
    path('get_analysis', AIAnalysis.as_view()),
]
