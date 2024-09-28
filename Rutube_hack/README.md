# Rutube_hack документация
Данный проект представляет собой Django проект с установленным Django Rest Framework (DRF), необходимом для общения данного бекенда с Front`end проектом с помощью REST.
Также было создано приложение ai, в котором происходит обработка GET-запросов.

Приложение ai, как и приложение rest_framework, являющееся внутренним django приложением для библиотеки DRF, были добавлены в список INSTALLED_APPS файла setting.py, описывающего основные настройки Django проекта. Без добавления данных наименований в список INSTALLED_APPS они не будут интегрированы в проект полноценно.

В файле urls.py базового приложения Rutube_hack был подключен используемый в приложении ai файл urls.py. Все адреса из файла urls.py приложения ai будут иметь "ai/" в своем пути. Сам файл urls.py приложения ai описывает подключаемый class-based view отработки приложения AIAnalysis по адресу /ai/get_analysis/. 
Класс view AIAnalysis описан в файле views.py приложения ai и представляет собой наследуемый от APIView класс, предназначенный для получения GET запроса с данными о тегируемом видеоролике и отправки ответа с полученными от нейросети тегами.
Метод get данного класса вызывается, когда по адресу /ai/get_analysis/ отправляется get-запрос. Вне класса подключаются модель и токенизатор, устанавливается работа CUDA-ядер, если они есть. В самом методе формируется ответ на запрос с помощью подключенной нейросети. Данные из тела запроса передаются на модель, которая выдает теги. Они, в свою очередь, формируются в словарь и отправляются как json.

Все остальные файлы и их содержимое генерируется автоматически при создании Django-проекта