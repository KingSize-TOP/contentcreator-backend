from fastapi import FastAPI
from googleapiclient.discovery import build
import os
import re
import math
from pydub import AudioSegment
from google.cloud import speech
import yt_dlp as youtube_dl
import time
import requests
import openai

app = FastAPI()

api_key = os.getenv('YOUTUBE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
heygen_key = os.getenv('HEYGEN_API_KEY')

if not api_key or not openai_api_key or not heygen_key:
    raise HTTPException(status_code=500, detail="API keys are not configured properly")

def get_channel_id_from_username(api_key, username):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        part='snippet',
        q=username,
        type='channel',
        maxResults=1
    )
    response = request.execute()
    if response['items']:
        return response['items'][0]['id']['channelId']
    else:
        raise ValueError("Channel not found for the given username.")

def get_videos_from_youtube_channel(api_key, profile_url):
    match = re.match(r'https://www\.youtube\.com/@([^/]+)', profile_url)
    if match:
        username = match.group(1)
    else:
        raise ValueError("Invalid YouTube profile URL format.")

    channel_id = get_channel_id_from_username(api_key, username)
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    )
    response = request.execute()
    playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    videos = []
    request = youtube.playlistItems().list(
        part='snippet',
        playlistId=playlist_id,
        maxResults=50
    )

    while request:
        response = request.execute()
        for item in response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            video_title = item['snippet']['title']
            videos.append({'video_id': video_id, 'title': video_title})

        request = youtube.playlistItems().list_next(request, response)

    return videos

def get_video_statistics(api_key, video_ids):
    youtube = build('youtube', 'v3', developerKey=api_key)
    stats = {}

    # Process video IDs in batches of 50
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i + 50]
        request = youtube.videos().list(
            part='statistics',
            id=','.join(batch_ids)
        )
        response = request.execute()
        for item in response['items']:
            video_id = item['id']
            view_count = int(item['statistics'].get('viewCount', 0))
            like_count = int(item['statistics'].get('likeCount', 0))
            stats[video_id] = {'views': view_count, 'likes': like_count}
    return stats

def get_top_videos(api_key, profile_url, top_n=5):
    videos = get_videos_from_youtube_channel(api_key, profile_url)
    video_ids = [video['video_id'] for video in videos]
    stats = get_video_statistics(api_key, video_ids)

    for video in videos:
        video_id = video['video_id']
        video['views'] = stats[video_id]['views']
        video['likes'] = stats[video_id]['likes']

    # Sort videos based on views and likes (weigh views more heavily here, adjust as needed)
    videos.sort(key=lambda x: (x['views'], x['likes']), reverse=True)

    return videos[:top_n]

def download_audio(video_url, output_format='wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': output_format,
            'preferredquality': '192'
        }],
        'outtmpl': 'audio.%(ext)s',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return f"audio.{output_format}"

def split_audio(audio_file_path, chunk_length_ms=30000):
    # Load audio file
    audio = AudioSegment.from_wav(audio_file_path)
    # Set audio to mono and 16 kHz for consistency
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_file, format="wav")
        chunks.append(chunk_file)
    return chunks

def transcribe_audio_google(audio_file_path):
    client = speech.SpeechClient()

    # Load audio into memory
    with open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)

    transcription = ''
    for result in response.results:
        transcription += result.alternatives[0].transcript + '\n'
    return transcription

def process_video(video_url):
    audio_file_path = download_audio(video_url)
    audio_chunks = split_audio(audio_file_path)

    full_transcription = ''
    for chunk in audio_chunks:
        try:
            transcription = transcribe_audio_google(chunk)
            full_transcription += transcription
        except Exception as e:
            print(f"Error transcribing chunk {chunk}: {e}")
        finally:
            os.remove(chunk)  # Clean up the chunk file

    os.remove(audio_file_path)  # Clean up the original audio file
    return full_transcription

def generate_similar_text(transcription, openai_api_key):
    openai.api_key = openai_api_key

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" or another available model
        messages=[
            {"role": "system", "content": "I want to generate similar text."},
            {"role": "user", "content": transcription}
        ],
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].message.content

def generate_avatar(transcription, heygen_key):
    gen_avatar_url = "https://api.heygen.com/v2/video/generate"
    headers = {"accept": "application/json", "content-type": "application/json", "x-api-key": heygen_key}
    payload = {
        "caption": False,
        "title": "string",
        "callback_id": "string",
        "dimension": {
            "width": 1280,
            "height": 720
        },
        "video_inputs": [
        {
            "character": {
                "type": "avatar",
                "avatar_id": "Gerardo_standing_outdoorsport_side",
                "scale": 1,
                "avatar_style": "normal",
                "offset": {
                  "x": 0,
                  "y": 0
                }
            },
            "voice": {
                "type": "text",
                "voice_id": "1985984feded457b9d013b4f6551ac94",
                "input_text": transcription
            },
            "background": {
                "type": "color",
                "value": "#f6f6fc"
            }
        }
        ],
        "callback_url": "string"
    }
    response = requests.post(gen_avatar_url, json=payload, headers=headers)

    return response.json()

def check_video_status(heygen_key, video_id):
    status_url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    headers = {"accept": "application/json", "x-api-key": heygen_key}
    while True:
        response = requests.get(status_url, headers=headers)
        
        if response.status_code == 200:
            status_info = response.json()
            status = status_info.get('data').get('status')
            
            if status == "completed":
                video_url = status_info.get('data').get('video_url')  # Assume this is the key for the video URL
                print("Video generation completed.")
                return video_url
            elif status == "processing":
                print("Video is still processing. Checking again in 30 seconds...")
                time.sleep(30)  # Wait before checking again
            elif status == "waiting":
                print("Video is still waiting. Checking again in 30 seconds...")
                time.sleep(30)  # Wait before checking again
            elif status == "pending":
                print("Video is still pending. Checking again in 30 seconds...")
                time.sleep(30)  # Wait before checking again
            else:
                raise Exception(f"Video generation failed or encountered an unknown status: {status}")
        else:
            raise Exception(f"Failed to check video status. Status: {response.status_code}, Response: {response.text}")

def download_video(video_url, output_filename):
    response = requests.get(video_url)
    
    if response.status_code == 200:
        with open(output_filename, 'wb') as video_file:
            video_file.write(response.content)
        print(f"Video downloaded successfully as {output_filename}.")
    else:
        raise Exception(f"Failed to download video. Status: {response.status_code}, Response: {response.text}")


@app.get("/videos")
def get_videos(profile_url: str):
    top_videos = get_top_videos(api_key, profile_url, 5)
    return top_videos

@app.get("/transcript_video")
def get_videos(video_id: str):
    youtube_url = f"https://www.youtube.com/watch?v={video['video_id']}"
    transcription = process_video(youtube_url)
    return transcription

@app.get("/generate_text")
def generate_text(transcription: str):
    similar_text = generate_similar_text(transcription, openai_api_key)
    return similar_text

@app.get("/generate_video")
def generate_video(text: str):
    response = generate_avatar(text, heygen_key)
    video_id = response.get("data").get("video_id")
    video_url = check_video_status(heygen_key, video_id)
    return video_url