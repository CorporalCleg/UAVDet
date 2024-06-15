import streamlit as st
import os
from ultralytics import YOLO
from VideoProcessor import MediaProcessor, process_media

# Создание папок для загрузки и обработки файлов
def create_folders(upload_folder="uploaded_files", processed_folder="processed_files"):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

# Функция для загрузки файлов
def save_uploaded_file(uploaded_file, folder_name="uploaded_files"):
    file_path = os.path.join(folder_name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Функция для отображения файлов с центровкой
def display_file(selected_file, folder_name="processed_files"):
    file_path = os.path.join(folder_name, selected_file)
    if selected_file.endswith('.mp4'):
        st.video(file_path)
    else:
        st.image(file_path, use_column_width=True)

def exclude_processed_files(file_list, processed_files):
    #print(f'Уже обработанные файлы: {processed_files}')
    #print(f'basename: {os.path.basename(file_list[0].name)} или {file_list[0].name}')
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
    #print(f'uploaded_files: {uploaded_files}')
    if uploaded_files:
        input_paths = []
        # Исключение уже обработанных файлов
        #print(st.session_state.processed_files)
        new_files = exclude_processed_files(uploaded_files, st.session_state.processed_files)
        #print(f'new_files: {new_files}')
        for uploaded_file in new_files:
            file_path = save_uploaded_file(uploaded_file)
            input_paths.append(file_path)
        if input_paths:
            st.toast(f"Файлы загружены", icon="🟢")
            imgs, vids = process_media(input_paths, processor)
            #print(f'input_paths: {input_paths}')
            # Получение реальных названий файлов с расширениями, но без папки
            new_variants = [os.path.basename(i) for i in imgs + vids]
            st.session_state.variants.extend(new_variants)
            st.toast(f"Файлы обработаны", icon="🟢")

            # Добавление обработанных файлов в processed_files
            st.session_state.processed_files.extend([os.path.basename(i.file_id) for i in new_files])
    
    # Удаление дубликатов из вариантов
    st.session_state.variants = list(set(st.session_state.variants))

    # Поле для выбора файла из выпадающего списка
    if st.session_state.variants:
        selected_file = st.selectbox("Выберите файл", st.session_state.variants)
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
    model_path = 'trained_y8m.pt'  # Укажите путь к модели
    processor = MediaProcessor('processed_files', model_path, batch_size=16)

    main(processor)