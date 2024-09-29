# Документация
## Используемые библиотеки
- streamlit – для создания веб-интерфейса.
- requests – для выполнения HTTP-запросов.
- base64 – для кодирования видеофайла в Base64.
- os – для работы с переменными окружения.
- dotenv – для загрузки переменных окружения из .env файла.

## Описание работы


1. Переменные окружения: **load_dotenv()** – загружает IP и порт сервера из .env файла. Это позволяет динамически настраивать адрес сервера, к которому приложение отправляет запросы.
os.getenv("IP") и os.getenv("PORT") – используются для формирования URL сервера.


2. Стилизация интерфейса: **st.markdown(page_bg_img, unsafe_allow_html=True)** – применяется для задания фона страницы и настройки внешнего вида с помощью CSS.
В CSS используются селекторы, такие как _[data-testid="stAppViewContainer"]_ для изменения фона и цвета элементов.


3. Текстовые поля: **st.text_input('Введите название видео')** – создает поле для ввода заголовка видео.
Загрузка видео: **st.text_area('Введите описание видео')** – позволяет пользователю ввести описание видео в многострочном поле.


4. **st.file_uploader('Перетащите видео файл', type=["mp4", "avi", "mov"])** – создает элемент для загрузки видеофайлов.
**uploaded_video.read()** – считывает содержимое загруженного файла.
**base64.b64encode(video_content)** – преобразует содержимое видео в строку формата Base64.

5. Отправка данных:
 **data_to_send** – объект JSON, содержащий video_title, video_description и закодированное видео.
**requests.get(address, json=data_to_send)** – отправляет запрос на сервер для анализа видео.

6. Обработка ответа:
 **get_response.json()** – получает данные с сервера (теги и вероятности).
**st.write(f"• {tag}: {prob}")** – выводит теги и их вероятности в интерфейсе.
