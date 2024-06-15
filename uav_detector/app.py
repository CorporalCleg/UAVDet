import streamlit as st
import os, zipfile, io
from ultralytics import YOLO
from VideoProcessor import MediaProcessor, process_media
import pandas as pd

metadata_folder = 'uav_detector/metadata'
processed_files_folder = 'uav_detector/processed_files'
uploaded_files_folder = 'uav_detector/uploaded_files'
model_folder = 'uav_detector/models'

# Создание папок для загрузки и обработки файлов
def create_folders(upload_folder=uploaded_files_folder, processed_folder=processed_files_folder):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

# Функция для загрузки файлов
def save_uploaded_file(uploaded_file, folder_name=uploaded_files_folder):
    file_path = os.path.join(folder_name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def zip_files(metadata_folder, file_paths):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            # Получаем базовое имя файла без расширения и добавляем .csv
            pure_name = os.path.splitext(os.path.basename(file_path))[0]
            full_path = os.path.join(metadata_folder, pure_name + '.csv')
            print(full_path)
            zipf.write(full_path, arcname=pure_name + '.csv')
    zip_buffer.seek(0)
    return zip_buffer

# Функция для отображения файлов с центровкой
def display_file(selected_file, folder_name=processed_files_folder):
    file_path = os.path.join(folder_name, selected_file)
    if selected_file.endswith('.mp4'):
        print(file_path)
        st.video(file_path)
        #table = pd.read_csv('metadata/drones.mp4_detection_results.csv')
        #table.reset_index(drop=True, inplace=True)
        #table.rename_axis('id', axis='index', inplace=True)
        #table.reset_index(inplace=True)
        #table['timestamp'] = (table['timestamp']*1000)#.astype('int')
        #table = table[['id', 'timestamp', 'class']]
        #items = table.to_dict(orient='records')
        #timeline = st_timeline(items, groups=[], options=options, height="300px")
    else:
        st.image(file_path, use_column_width=True)

def exclude_processed_files(file_list, processed_files):
    return [file for file in file_list if os.path.basename(file.file_id) not in processed_files]

# Основная функция приложения
def main(processor):
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'variants' not in st.session_state:
        st.session_state.variants = []

    # Создание папок для загрузки и обработки файлов
    create_folders()

    # Заголовок приложения
    st.title("Загрузите фото и видео, затем выберите файл из списка")

    # Загрузка файлов
    uploaded_files = st.file_uploader("Загрузите фото и видео", accept_multiple_files=True)
    if uploaded_files:
        input_paths = []
        new_files = exclude_processed_files(uploaded_files, st.session_state.processed_files)
        for uploaded_file in new_files:
            file_path = save_uploaded_file(uploaded_file)
            input_paths.append(file_path)
        if input_paths:
            st.toast(f"Файлы загружены", icon="🟢")
            imgs, vids = process_media(input_paths, processor)
            new_variants = [os.path.basename(i) for i in imgs + vids]
            st.session_state.variants.extend(new_variants)
            st.toast(f"Файлы обработаны", icon="🟢")

            st.session_state.processed_files.extend([os.path.basename(i.file_id) for i in new_files])
    
    # Удаление дубликатов из вариантов
    st.session_state.variants = list(set(st.session_state.variants))

    # Контейнер для выпадающего меню и кнопки скачивания
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_file = st.selectbox("Выберите файл", st.session_state.variants, label_visibility='collapsed')
    with col2:
        zip_buffer = zip_files(metadata_folder, st.session_state.variants)
        st.download_button(
            label="Скачать zip",
            data=zip_buffer,
            file_name="files.zip",
            mime="application/zip"
        )

    # Центровка и отображение выбранного файла
    if selected_file:
        st.markdown(
            """
            <style>
            .centered {
                display: flex;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        display_file(selected_file)
        st.markdown('</div>', unsafe_allow_html=True)
        
# Запуск приложения
if __name__ == "__main__":
    model_path = f'{model_folder}/yolo8m_last.pt'  # Укажите путь к модели
    processor = MediaProcessor(processed_files_folder, model_path, batch_size=16)

    main(processor)
