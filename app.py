import cv2
import base64
import os
import requests
from openai import OpenAI
from collections import deque
import pygame
from io import BytesIO
from datetime import datetime

def encode_image_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode('utf-8')

def send_frame_to_gpt(frame, previous_texts, client):
    context = ' '.join(previous_texts)
    prompt_message = f"Context: {context}. Explain what you see briefly and concisely. Describe what you see at that moment. Pay attention to objects Give short and concise answers. Quick, short and concise answers. Briefly write down what you see instantly.."

    PROMPT_MESSAGES = {
        "role": "user",
        "content": [
            prompt_message,
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
        ],
    }

    params = {
        "model": "gpt-4-vision-preview",
        "messages": [PROMPT_MESSAGES],
        "max_tokens": 500,
    }

    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
pygame.mixer.init()

video = cv2.VideoCapture(0)

previous_texts = deque(maxlen=5)

pause_processing = False

while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    base64_image = encode_image_to_base64(frame)

    if not pause_processing:
        generated_text = send_frame_to_gpt(base64_image, previous_texts, client)
        print(f"Generated Text: {generated_text}")

        # Use Elevenlabs API for text-to-speech
        CHUNK_SIZE = 1024
        url = "https://api.elevenlabs.io/v1/text-to-speech/X7JFHsSvB3SVTLssl4Jx"
       
        payload = {
            "text": generated_text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": "<apikey>"
        }

        response = requests.post(url, json=payload, headers=headers)


        audio_data = BytesIO(response.content)

 
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()

        print(response.text)

   


video.release()
pygame.mixer.quit()
