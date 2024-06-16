# Приложение для поиска летательных аппаратов на фото и видео

Проект представляет из себя интерфейс для работы с предварительно обученной моделью Yolo 8. Модель Yolo 8 была обучена на поиск беспилотников разных типов, а также птиц, самолётов и вертолётов 🐦  ✈️ 🚁

## Возможности

- Поиск объектов на видео и фото
- Поддерживаемые расширения: mp4, mov, jpg, jpeg, png, bmp, tiff, heic, heif, webp
- Выгрузка файлов с данными о результатах распознавания

## Требования

- Python 3.11

## Запуск через poetry

Склонируйте этот репозиторий на ваш локальный компьютер:

```bash
git clone https://github.com/CorporalCleg/UAVDet.git
```

Перейдите в каталог проекта:

```bash
cd file-extension-validator
```

Инициализируйте виртуальное окружение poetry
```bash
poetry install
```

Запустите приложение
```bash
poetry run streamlit run uav_detector/app.py
```
## Запуск в контейнере(Docker)

### Утстановка nvidia-container-toolkit
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Сборка образа
```bash
docker build -t uav_test .
```

### Запуск контейнера
```bash
docker run -it --rm --privileged -p 8501:8501 --gpus all uav_test streamlit run uav_detector/app.py

```

После запуска приложения вы увидите браузерный интерфейс. В этом руководстве объясняется, как загружать и обрабатывать файлы, а также как скачивать результаты обработки.

## Загрузка файлов

1. **Перетаскивание файлов**: В поле для ввода файлов можно перетаскивать фотографии или видео.
2. **Ожидание загрузки**: После загрузки файлов дождитесь сообщений "Файлы загружены" и "Файлы обработаны" в правом верхнем углу.

## Выбор и обработка файлов

1. **Выпадающее меню**: После появления уведомлений "Файлы загружены" и "Файлы обработаны", станет доступно выпадающее меню.
2. **Выбор файла**: В выпадающем меню выберите необходимый файл. Файл будет выведен с ограничивающими рамками.

## Скачивание обработанных файлов

### Изображения

- Обработанные изображения всегда сохраняются с расширением `.png`, независимо от изначального расширения.
- Для скачивания изображений:
  1. Нажмите правой кнопкой мыши на изображение.
  2. Выберите "Сохранить как".

### Видео

- Обработанные видео сохраняют исходные расширения.
- Для скачивания видео:
  1. Нажмите на троеточие в правом нижнем углу проигрывателя.
  2. Найдите и нажмите кнопку "Скачать".

## Скачивание результатов распознавания объектов

### Текстовые файлы

- Для фотографий создаются текстовые файлы, в которых построчно записаны найденные объекты с указанием класса объекта, координат и размеров рамки.

### CSV файлы

- Для видео создаются файлы формата CSV, которые содержат информацию о распознанных объектах:
  - Класс объекта
  - Координаты и размеры рамки
  - Номер фрейма, на котором был распознан объект
  - Время, рассчитанное как `номер фрейма/fps` исходного видео

## Скачивание ZIP архива

- Вы можете скачать ZIP архив со всеми таблицами и текстовыми файлами:
  1. Нажмите на кнопку "Скачать ZIP".
  2. Архив со всеми загруженными вами файлами с последней перезагрузки страницы скачается на ваш компьютер.

## Добавление другой модели

Для того, чтобы использовать другую модель нужно положить веса модели в директорию `uav_detector/models`

Затем изменить в файле `uav_detector/app.py`

```python
...
model = 'name_of_model.pt'
...
```
Далее после повтроной сборки образа приложение заработает с новой моделью