import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import pillow_heif
import numpy as np

class MediaProcessor:
    def __init__(self, output_folder, model_path, confidence_threshold=0.25, batch_size=16):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_folder = output_folder
        self.metadata_folder = 'uav_detector/metadata' 
        self.model = YOLO(model_path).to(self.device)
        self.confidence_threshold = confidence_threshold
        self.classes = self.model.names
        self.batch_size = batch_size

    def load_image(self, path):
        if path.lower().endswith('.heic'):
            heif_file = pillow_heif.open_heif(path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            return cv2.imread(path)

    def get_Boxes_and_Tables(self, pics):
        annotated_images = []
        tables = []
        for pic in pics:
            # Фильтруем результаты на основе порога уверенности
            #high_conf_indices = pic.boxes.conf > confidence_threshold

            #filtered_boxes = pic.boxes[high_conf_indices]

            # Обновляем данные в объекте `Boxes`
            #pic.boxes = filtered_boxes
            
            # Создаем DataFrame из отфильтрованных данных
            data_dict = {
                'class': pic.boxes.cls.cpu().numpy(),
                'confidence': pic.boxes.conf.cpu().numpy(),
                'x_center': pic.boxes.xywh[:, 0].cpu().numpy(),
                'y_center': pic.boxes.xywh[:, 1].cpu().numpy(),
                'width': pic.boxes.xywh[:, 2].cpu().numpy(),
                'height': pic.boxes.xywh[:, 3].cpu().numpy()
            }
            df = pd.DataFrame(data_dict)
            tables.append(df)

            # Используем метод plot для получения изображения с рамками
            annotated_image = pic.plot()
            annotated_images.append(annotated_image)


            # Поскольку OpenCV загружает изображения в формате BGR, преобразуем их в формат RGB для корректного отображения
            #annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return annotated_images, tables

    def save_pics(self, images, to_process_names, tables):
        os.makedirs(self.output_folder, exist_ok=True)

        save_paths = []
        # Сохраняем каждое изображение в отдельный файл
        for img_array, name_inp, table in zip(images, to_process_names, tables):
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            # Формируем новое имя
            name_out = os.path.splitext(os.path.basename(name_inp))[0]

            img = Image.fromarray(img_array)
            # Создаем корректный путь к файлу
            output_path = os.path.join(self.output_folder, f'{name_out}.png')
            img.save(output_path)
            output_path_metadata = os.path.join(self.metadata_folder, f'{name_out}.csv')
            table.to_csv(output_path_metadata)
            save_paths.append(output_path)
        return save_paths

    def process_images(self, to_process_names):
        #results = self.model(images, verbose=False, save_dir='processed_files', save=True)
        data = self.model(to_process_names, conf = self.confidence_threshold, batch=self.batch_size)
        pics, tables = self.get_Boxes_and_Tables(data)
        save_paths = self.save_pics(pics, to_process_names, tables)

        return save_paths

    def process_single_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        output_video_path = os.path.join(self.output_folder, os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')#*'avc1')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        frame_num = 0
        # Инициализация пустого тензора для хранения результатов
        all_classes = []
        all_confidences = []
        all_x_centers = []
        all_y_centers = []
        all_widths = []
        all_heights = []
        frame_nums = []

        with tqdm(total=total_frames, desc=f"Processing Video {os.path.basename(video_path)}", position=0, leave=True) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)
                frame_num += 1

                if len(frames) == self.batch_size or frame_num == total_frames:
                    results = self.model(frames, verbose=False, conf=0.25)

                    for i, result in enumerate(results):
                        num_detections = len(result.boxes.cls)
                        if num_detections > 0:
                            all_classes.extend(result.boxes.cls.cpu().tolist())
                            all_confidences.extend(result.boxes.conf.cpu().tolist())
                            all_x_centers.extend(result.boxes.xywh[:, 0].cpu().tolist())
                            all_y_centers.extend(result.boxes.xywh[:, 1].cpu().tolist())
                            all_widths.extend(result.boxes.xywh[:, 2].cpu().tolist())
                            all_heights.extend(result.boxes.xywh[:, 3].cpu().tolist())
                            frame_nums.extend([frame_num + i] * num_detections)

                        annotated_frame = result.plot()  # Получаем кадр с нанесенными рамками
                        out.write(annotated_frame)

                    frames = []
                    pbar.update(self.batch_size)
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        #print(len(frame_nums), len(all_classes), len(all_x_centers), len(all_y_centers), len(all_widths), len(all_heights))
        # Создание DataFrame

        # Перевод тензоров на CPU и преобразование в списки
        data = {
            'frame_num': frame_nums,
            'class': all_classes,
            'x_center': all_x_centers,
            'y_center': all_y_centers,
            'width': all_widths,
            'height': all_heights
        }

        tab = pd.DataFrame(data)
        tab['class'] = tab['class'].astype(int)#.map(self.classes)
        os.makedirs(self.metadata_folder, exist_ok=True)
        tab.to_csv(os.path.join(self.metadata_folder, f'{os.path.splitext(os.path.basename(video_path))[0]}.csv'))
        return output_video_path

    def process_videos(self, video_paths):
        vids = []
        for video_path in video_paths:
            output_video_path = self.process_single_video(video_path)
            vids.append(output_video_path)
        return vids

def process_media(input_paths, processor):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.heic', '.heif', '.webp')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

    image_paths = [path for path in input_paths if path.lower().endswith(image_extensions)]
    video_paths = [path for path in input_paths if path.lower().endswith(video_extensions)]

    img_save_paths_list, vid_save_paths_list = [], []

    if image_paths:
        img_save_paths_list = processor.process_images(image_paths)
    if video_paths:
        vid_save_paths_list = processor.process_videos(video_paths)
    print(vid_save_paths_list)
    return img_save_paths_list, vid_save_paths_list
