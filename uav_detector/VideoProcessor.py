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
    def __init__(self, output_path, model_path, batch_size=16):
        self.output_path = output_path
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.model_path).to(self.device)
        self.colors = {
            0: (255, 0, 0),    # quadrotor - красный
            1: (0, 255, 0),    # airplane - зеленый
            2: (0, 0, 255),    # helicopter - синий
            3: (255, 255, 0),  # bird - желтый
            4: (255, 0, 255)   # uav-plane - фиолетовый
        }
        self.batch_size = batch_size

    def process_single_video(self, video_path):
        print("----")
        print('----')
        print(video_path)
        print("----")
        print('----')
        cap = cv2.VideoCapture(video_path)
        output_video_path = os.path.join(self.output_path, os.path.basename(video_path))
        print("----")
        print('----')
        print(output_video_path)
        print("----")
        print('----')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')#*'avc1')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        print('OK')
        print('OK')
        print('OK')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        columns = ['frame_num', 'timestamp', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2']
        data = []

        frame_num = 0
        
        while cap.isOpened():
            ret, frame_one = cap.read()
            
            if ret:
                frame_num += 1
                result = self.model(frame_one, verbose=False)[0]

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    data.append([frame_num, 0, self.model.names[int(cls)], conf, int(x1), int(y1), int(x2), int(y2)])

                out.write(result.plot())
            else:
                break

        cap.release()
        out.release()

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join('uav_detector/metadata', f"{os.path.basename(video_path)}_detection_results.csv"), index=False)
        print(df)
        return output_video_path

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

    def process_images(self, input_paths):
        images = [self.load_image(path) for path in input_paths]
        results = self.model(images, verbose=False)
        processed_images = []

        for i, result in enumerate(results):
            processed_image_path = os.path.join(self.output_path, os.path.basename(input_paths[i]).split('.')[0] + '.png')
            processed_images.append(processed_image_path)
            result.save(processed_image_path)

        return processed_images

    def process_videos(self, input_paths):
        vids = []
        for video_path in input_paths:
            output_video_path = self.process_single_video(video_path)
            vids.append(output_video_path)
        return vids

def process_media(input_paths, processor):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.heic', '.heif', '.webp')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

    image_paths = [path for path in input_paths if path.lower().endswith(image_extensions)]
    video_paths = [path for path in input_paths if path.lower().endswith(video_extensions)]

    imgs, vids = [], []

    if image_paths:
        imgs = processor.process_images(image_paths)
    if video_paths:
        vids = processor.process_videos(video_paths)
    return imgs, vids
