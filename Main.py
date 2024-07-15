import cv2
import requests
import numpy as np
import mediapipe as mp
from fer import FER
import gradio as gr
import time
from facenet_pytorch import MTCNN
from deepface import DeepFace


class DeepFaceEmotionDetector:
    def __init__(self):
        pass

    def detect_emotions(self, image):
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotions = result[0]['emotion']
        return [{"emotions": emotions}]

# Инициализация детекторов
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=10, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
fer_detector = FER(mtcnn=True)
deepface_emotion_detector = DeepFaceEmotionDetector()
mtcnn_detector = MTCNN(keep_all=True, margin=30)


def load_image_from_url(url):
    try:
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def get_bounding_box(face_landmarks, image_shape):
    x_min, x_max = image_shape[1], 0
    y_min, y_max = image_shape[0], 0
    for landmark in face_landmarks.landmark:
        x, y = int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])
        x_min, x_max = min(x, x_min), max(x, x_max)
        y_min, y_max = min(y, y_min), max(y, y_max)
    return (x_min, y_min), (x_max, y_max)

def convert_to_rgb(image): #todo: Убрать эту функцию
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_faces_mediapipe(image_rgb):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=10, min_detection_confidence=0.1, min_tracking_confidence=0.95
    )
    return face_mesh.process(image_rgb)

def detect_faces_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def detect_faces_mtcnn(image_rgb):
    boxes, _ = mtcnn_detector.detect(image_rgb)
    return boxes

def draw_face_boxes(image, faces, color, face_detector_choice):
    if face_detector_choice == "MediaPipe":
        for face_landmarks in faces:
            bbox = get_bounding_box(face_landmarks, image.shape)
            cv2.rectangle(image, bbox[0], bbox[1], color, 2)
    elif face_detector_choice == "OpenCV":
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    elif face_detector_choice == "MTCNN":
        for face in faces:
            x, y, width, height = face
            cv2.rectangle(image, (int(x), int(y)), (int(width), int(height)), color, 2)
    return image

def extract_faces(image_rgb, faces, face_detector_choice):
    face_list = []
    if face_detector_choice == "MediaPipe":
        for face_landmarks in faces:
            bbox = get_bounding_box(face_landmarks, image_rgb.shape)
            face = image_rgb[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            face_list.append((face, bbox))
    elif face_detector_choice == "OpenCV":
        for (x, y, w, h) in faces:
            face = image_rgb[y:y + h, x:x + w]
            bbox = ((x, y), (x + w), (y + h))
            face_list.append((face, bbox))
    elif face_detector_choice == "MTCNN":
        for face in faces:
            x, y, width, height = face
            face = image_rgb[int(y):int(y + height), int(x):int(x + width)]
            bbox = ((int(x), int(y)), (int(x + width), int(y + height)))
            face_list.append((face, bbox))
    return face_list

def detect_emotions_on_faces(faces, emotion_detector):
    emotions_list = []
    for face, bbox in faces:
        if face.size != 0:
            emotions = emotion_detector(face)
            if emotions:
                emotions_list.append((emotions, bbox))
            else:
                emotions_list.append((None, bbox))
        else:
            emotions_list.append((None, bbox))
    return emotions_list

def draw_labels(image, emotions_list, color):
    for i, (_, bbox) in enumerate(emotions_list):
        text_x = bbox[0][0]
        text_y = bbox[0][1] - 10 if bbox[0][1] - 10 > 10 else bbox[1][1] + 20
        cv2.putText(image, f"Face {i + 1}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return image

def process_image(image, face_detector_choice, emotion_detector_choice, color):
    image_rgb = convert_to_rgb(image)
    start_time = time.time()

    if face_detector_choice == "MediaPipe":
        results = detect_faces_mediapipe(image_rgb)
        faces = results.multi_face_landmarks if results.multi_face_landmarks else []
    elif face_detector_choice == "OpenCV":
        faces = detect_faces_opencv(image)
    elif face_detector_choice == "MTCNN":
        faces = detect_faces_mtcnn(image_rgb)
    else:
        return None, "Face detector not supported", 0, 0

    end_time_faces = time.time()
    face_detection_time = end_time_faces - start_time

    if emotion_detector_choice == "FER":
        emotion_detector = fer_detector.detect_emotions
    elif emotion_detector_choice == "DeepFace":
        emotion_detector = deepface_emotion_detector.detect_emotions
    else:
        return None, "Emotion detector not supported", 0, 0

    faces_for_emotion_detection = extract_faces(image_rgb, faces, face_detector_choice)
    emotions_list = detect_emotions_on_faces(faces_for_emotion_detection, emotion_detector)
    end_time_emotions = time.time()
    emotion_detection_time = end_time_emotions - end_time_faces

    image = draw_face_boxes(image, faces, color, face_detector_choice)
    image = draw_labels(image, emotions_list, color)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, emotions_list, face_detection_time, emotion_detection_time

def process_input(image, url, face_detector_choice, emotion_detector_choice, color):
    if url:
        image = load_image_from_url(url)

    if image is not None:
        color_tuple = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        processed_image, emotions_list, face_detection_time, emotion_detection_time = process_image(
            image, face_detector_choice, emotion_detector_choice, color_tuple
        )

        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        emotions_text = []
        for i, (emotions, bbox) in enumerate(emotions_list):
            if emotions:
                sorted_emotions = sorted(emotions[0]['emotions'].items(), key=lambda x: x[1], reverse=True)
                emotion_labels = [f"{emotion.capitalize()}: {prob:.2f}" for emotion, prob in sorted_emotions]
                emotions_text.append(f"Face {i + 1} ({bbox}): " + ", ".join(emotion_labels))
            else:
                emotions_text.append(f"Face {i + 1} ({bbox}): No emotions detected")

        timing_text = f"Face Detection Time: {face_detection_time:.2f}s\nEmotion Detection Time: {emotion_detection_time:.2f}s"

        return processed_image_rgb, "\n".join(emotions_text), timing_text
    else:
        return None, "Failed to load image.", ""

# Интерфейс Gradio
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy", label="Upload Image")
            url_input = gr.Textbox(label="Or Enter Image URL")
            face_detector_choice_input = gr.Radio(choices=["MediaPipe", "OpenCV", "MTCNN"], label="Choose Face Detector")
            emotion_detector_choice_input = gr.Radio(choices=["FER", "DeepFace"], label="Choose Emotion Detector")
            color_input = gr.ColorPicker(value="#FF0000", label="Select Box Color")
            submit_button = gr.Button("Process")
        with gr.Column():
            img_output = gr.Image(label="Processed Image")
            text_output = gr.Textbox(label="Detected Emotions", lines=10)
            timing_output = gr.Textbox(label="Detection Timing", lines=2)

    submit_button.click(fn=process_input,
                        inputs=[img_input, url_input, face_detector_choice_input, emotion_detector_choice_input, color_input],
                        outputs=[img_output, text_output, timing_output])

demo.launch()
