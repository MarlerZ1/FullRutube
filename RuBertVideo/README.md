## RuBertOnlyVideo_Check.ipynb
Файл предполагает ручную проверку работы предсказания по описанию из видео.
- `model = AutoModelForSequenceClassification.from_pretrained("./saved_model")`: загружает предварительно обученную модель для классификации последовательностей из сохраненной директории `./saved_model`. Это модель на базе библиотеки `transformers`.
  
- `tokenizer = AutoTokenizer.from_pretrained("./saved_model")`: загружает токенизатор, связанный с моделью, из той же директории. Токенизатор используется для преобразования текста в формат, который понимает модель.

- `def normalize_text(text):`: функция для нормализации введенного текста. Она приводит текст к нижнему регистру, удаляет лишние пробелы и знаки препинания.

После нормализации выполняется предсказание по введенному описанию из видео и его вывод.
## RuBertOnlyVideoTrain.ipynb
В файле производится процесс обучения на основе описаний из видео
   - `model = AutoModelForSequenceClassification.from_pretrained("./saved_model")`: загружает предварительно обученную модель для классификации последовательностей из сохраненной директории `./saved_model`. Это модель на базе библиотеки `transformers`.
   - `def normalize_text(text)`: функция для нормализации введенного текста. Она приводит текст к нижнему регистру, удаляет лишние пробелы и знаки препинания.

Загружается файл ```'merged_data.csv'``` в котором представлены описания видео. Выборка разделяется на обучающую и тестовую. Обучается модель ```AutoModelForSequenceClassification``` и сохраняется в папку ```./saved_model```. Также сохраняется токенизатор и классы кодировщика меток. Результат предсказания тестового набора сохранен в файле ```"sample_submission.csv"```.


## VideoToText_CSV.ipynb

1. **Загрузка процессора и модели BLIP для описания изображений**
   - `processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")`: загружает процессор BLIP, который используется для предварительной обработки изображений перед подачей в модель.
   - `model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")`: загружает модель BLIP для генерации текстовых описаний изображений.

2. **Загрузка модели и токенизатора для перевода**
   - `translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")`: загружает модель для перевода текста с английского на русский.
   - `translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")`: загружает токенизатор, соответствующий модели перевода, для подготовки текста к переводу.

3. **Загрузка модели и токенизатора для суммаризации**
   - `summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')`: загружает модель BART для суммаризации текста.
   - `summarization_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')`: загружает токенизатор, связанный с моделью BART, для предварительной обработки текста перед суммаризацией.

5. **Извлечение кадров из видео**
   - `def extract_frames(video_path, num_frames=16)`: функция для извлечения кадров из видео. Извлекает определенное количество кадров (по умолчанию 16) из видеофайла.

6. **Генерация описаний для кадров с помощью BLIP**
   - `def generate_captions(frames, processor, model, device)`: функция, которая использует процессор и модель BLIP для генерации текстовых описаний каждого кадра видео. Используются сгенерированные описания для всех кадров и сохраняются уникальные.

7. **Суммаризация текста**
   - `def summarize_text(text, summarization_tokenizer, summarization_model)`: функция для суммаризации текста с помощью модели BART. Текст токенизируется и передается модели, которая возвращает краткую версию текста.

8. **Перевод текста на русский язык**
   - `def translate_to_russian(caption, translation_tokenizer, translation_model)`: функция для перевода текста на русский язык. Текст передается через токенизатор модели MarianMT, которая возвращает переведенный результат.

9. **Обработка видео и генерация переводов**
   - `def process_videos_and_generate_translations(video_folder, output_csv_path, processor, model, summarization_model, translation_model, device)`: основная функция, которая обрабатывает все видео в папке, генерирует описания для кадров, суммирует текст, переводит его на русский язык и записывает результаты в CSV файл.

10. **Объединение данных**
    - Данные из файла с переведенными описаниями видео загружаются с помощью `summarized_df = pd.read_csv('video_captions_translations.csv')`.
    - Затем они объединяются с другими данными на основе столбца `video_id`, и результат сохраняется в новый файл `merged_data.csv`.


