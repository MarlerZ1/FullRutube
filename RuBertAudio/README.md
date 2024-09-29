1. В файле ```Whisper_GetText.ipynb``` представлен процесс транскрипции аудиформата в текстовый.
Загружается модель Whisper small. Чтобы процесс обработки не занимал слишком много времени, размер видео скращается до первых 2.5 минут, а также учитываюся уже обработанные видео. Результат транскрипции сохраняется в файле ```"video_transcriptions.csv"```.
2. В файле ```Audio_GetSumm.ipynb``` представлен процесс суммаризации полученного текста.
Загружается модель Helsinki и BART. Так как модель BART лучше работает с английским текстом, необходимо было переводить из русского в английский перед суммаризацией. Считывается файл ```"video_transcriptions.csv"```, переводится в английский, суммаризируется, переводится в русский и записывается в файл ```"summarized_video_transcriptions.csv"```. Для экономии времени предусмотрен учет уже обработанных строк.
3. В файле ```RuBertOnlyAudioTrain.ipynb``` представлен процесс обучения на суммаризированном тексте.
Загружается модель RuBERT. Загружается файл ```'merged_data.csv'``` в котором представлен суммаризированный текст с тегами. Суммаризация нормализуется. Выборка разделяется на обучающую и тестовую. Обучается модель ```AutoModelForSequenceClassification``` и сохраняется в папку ```./saved_model```. Также сохраняется токенизатор и классы кодировщика меток. Результат предсказания тестового набора сохранен в файле ```"sample_submission.csv"```.
4. В файле ```RuBertOnlyAudio_Check.ipynb``` предполагается ручная проверка работы.
Загружается обученная модель и токенизатор из папки ```./saved_model```. Предоставляется поле для ввода суммаризированной транскрипции, после чего проводится суммаризация текста, загружается кодировщик меток и получение результата работы модели. Выводятся первые пять тегов.