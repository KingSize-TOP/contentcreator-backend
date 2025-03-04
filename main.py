from fastapi import FastAPI, HTTPException, Query
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
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
from datetime import timedelta
import isodate
from isodate import ISO8601Error
from hikerapi import Client

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Replace "*" with specific origins like ["http://localhost:3000"] for better security.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

api_key = os.getenv('YOUTUBE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
heygen_key = os.getenv('HEYGEN_API_KEY')
hiker_api_key = os.getenv('HIKER_API_KEY')

class GenerateVideoRequest(BaseModel):
    text: str
    avatar_id: str
    voice_id: str

if not api_key or not openai_api_key or not heygen_key:
    raise HTTPException(status_code=500, detail="API keys are not configured properly")

def convert_to_iso8601(duration_str):
    """
    Converts a human-readable duration string (e.g., '0:05:47') to ISO 8601 format (e.g., 'PT5M47S').
    """
    try:
        # Split the duration into hours, minutes, and seconds
        parts = duration_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = int(parts[0])
        else:
            raise ValueError(f"Invalid duration format: {duration_str}")

        # Construct the ISO 8601 duration string
        iso8601_duration = "PT"
        if hours > 0:
            iso8601_duration += f"{hours}H"
        if minutes > 0:
            iso8601_duration += f"{minutes}M"
        if seconds > 0:
            iso8601_duration += f"{seconds}S"
        return iso8601_duration
    except Exception as e:
        raise ValueError(f"Error converting duration to ISO 8601: {duration_str}. Error: {e}")

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

def get_video_language(api_key, video_id):
    """
    Fetches the default audio language of a YouTube video using the YouTube Data API.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Fetch video details
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()

    # Extract the default audio language from the video metadata
    if "items" in response and len(response["items"]) > 0:
        video = response["items"][0]
        default_language = video["snippet"].get("defaultAudioLanguage")  # Example: "en", "uk", "ru"
        return default_language
    else:
        return None        

def get_videos_from_youtube_channel(api_key, profile_url):
    # Extract username from profile URL
    match = re.match(r'https://www\.youtube\.com/@([^/]+)', profile_url)
    if match:
        username = match.group(1)
    else:
        raise ValueError("Invalid YouTube profile URL format.")

    # Get channel ID
    channel_id = get_channel_id_from_username(api_key, username)

    # Fetch channel details to get subscription count
    youtube = build('youtube', 'v3', developerKey=api_key)
    channel_request = youtube.channels().list(
        part='statistics',
        id=channel_id
    )
    channel_response = channel_request.execute()
    subscription_count = int(channel_response['items'][0]['statistics']['subscriberCount'])

    # Get the upload playlist ID
    playlist_request = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    )
    playlist_response = playlist_request.execute()
    playlist_id = playlist_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # Fetch videos from the playlist
    videos = []
    playlist_items_request = youtube.playlistItems().list(
        part='snippet',
        playlistId=playlist_id,
        maxResults=50
    )

    while playlist_items_request:
        playlist_items_response = playlist_items_request.execute()
        for item in playlist_items_response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            video_title = item['snippet']['title']
            video_thumbnail = item['snippet']['thumbnails']['high']['url']  # Get high-resolution thumbnail
            videos.append({
                'video_id': video_id,
                'title': video_title,
                'thumbnail': video_thumbnail
            })

        playlist_items_request = youtube.playlistItems().list_next(playlist_items_request, playlist_items_response)

    return {
        'subscription_count': subscription_count,
        'videos': videos
    }

# Function to fetch video statistics (views, likes) for multiple videos
def get_video_statistics(api_key, video_ids):
    youtube = build('youtube', 'v3', developerKey=api_key)
    stats = {}

    # YouTube API allows a maximum of 50 video IDs per request
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i + 50]
        request = youtube.videos().list(
            part='statistics, contentDetails',
            id=','.join(batch_ids)
        )
        response = request.execute()

        # Extract views and likes for each video
        for item in response.get('items', []):
            video_id = item['id']
            duration_iso = item['contentDetails'].get('duration', None)
            try:
                duration = isodate.parse_duration(duration_iso) if duration_iso else timedelta(seconds=0)
            except ISO8601Error:
                duration = timedelta(seconds=0)  # Default to 0 seconds if parsing fails
            stats[video_id] = {
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0)),
                'duration': str(duration)
            }
    return stats

# Function to fetch the top N videos from a channel
def get_sorted_videos(api_key, profile_url):
    # Fetch videos and subscription count
    channel_data = get_videos_from_youtube_channel(api_key, profile_url)
    videos = channel_data['videos']
    subscription_count = channel_data['subscription_count']

    # Get video statistics (views and likes)
    video_ids = [video['video_id'] for video in videos]
    stats = get_video_statistics(api_key, video_ids)

    # Add views, likes, and duration to each video
    filtered_videos = []
    for video in videos:
        video_id = video['video_id']
        video['views'] = stats.get(video_id, {}).get('views', 0)
        video['likes'] = stats.get(video_id, {}).get('likes', 0)
        video['duration'] = stats.get(video_id, {}).get('duration', '0:00:00')

        # Try parsing the duration and filter videos less than 2 minutes
        try:
            duration_str = video['duration']
            # Convert non-ISO 8601 durations (e.g., 0:05:47) to ISO 8601 if needed
            if not duration_str.startswith("P"):
                duration_str = convert_to_iso8601(duration_str)
            duration = isodate.parse_duration(duration_str)

            if duration < timedelta(minutes=2):
                filtered_videos.append(video)
        except (isodate.ISO8601Error, ValueError, TypeError) as e:
            # Log invalid durations and skip the video
            print(f"Skipping video ID {video_id} due to invalid duration: {video['duration']}, Error: {e}")
            continue

    # Sort videos based on views and likes (primary = views, secondary = likes)
    filtered_videos.sort(key=lambda x: (x['views'], x['likes']), reverse=True)

    return filtered_videos

def get_short_videos(api_key, profile_url):
    """
    Fetches only short videos (less than 60 seconds in duration) from a YouTube channel.
    """
    # Fetch videos and subscription count
    channel_data = get_videos_from_youtube_channel(api_key, profile_url)
    videos = channel_data['videos']

    # Get video statistics (views, likes, duration)
    video_ids = [video['video_id'] for video in videos]
    stats = get_video_statistics(api_key, video_ids)

    # Add views, likes, and duration to each video, and filter for short videos
    short_videos = []
    for video in videos:
        video_id = video['video_id']
        video['views'] = stats.get(video_id, {}).get('views', 0)
        video['likes'] = stats.get(video_id, {}).get('likes', 0)
        video['duration'] = stats.get(video_id, {}).get('duration', '0:00:00')  # Default to 0:00:00 if duration is missing

        # Try parsing the duration and filter for videos less than 60 seconds
        try:
            duration_str = video['duration']
            # Convert non-ISO 8601 durations (e.g., 0:05:47) to ISO 8601 if needed
            if not duration_str.startswith("P"):
                duration_str = convert_to_iso8601(duration_str)
            duration = isodate.parse_duration(duration_str)

            # Check if the duration is less than 60 seconds
            if duration < timedelta(seconds=60):
                short_videos.append(video)
        except (isodate.ISO8601Error, ValueError, TypeError) as e:
            # Log invalid durations and skip the video
            print(f"Skipping video ID {video_id} due to invalid duration: {video['duration']}, Error: {e}")
            continue

    # Sort short videos based on views and likes (primary = views, secondary = likes)
    short_videos.sort(key=lambda x: (x['views'], x['likes']), reverse=True)

    return short_videos

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

def transcribe_audio_google(audio_file_path, language_code="en-US"):
    client = speech.SpeechClient()

    # Load audio into memory
    with open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)
    print(response)

    transcription = ''
    for result in response.results:
        transcription += result.alternatives[0].transcript + '\n'
    # audio_file = open(audio_file_path, 'rb')
    # transcription = openai.audio.transcriptions.create(
    #     model="whisper-1",
    #     file=audio_file
    # )
    return transcription

def transcribe_audio_whisper(audio_file_path, openai_api_key):
    transcription = ''
    audio_file = open(audio_file_path, 'rb')
    openai.api_key = openai_api_key
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    audio_file.close()
    return transcription.text

def process_video(video_url, language_code="en-US"):
    audio_file_path = download_audio(video_url)
    audio_chunks = split_audio(audio_file_path)

    full_transcription = ''
    for chunk in audio_chunks:
        try:
            transcription = transcribe_audio_google(chunk, language_code)
            full_transcription += transcription
        except Exception as e:
            print(f"Error transcribing chunk {chunk}: {e}")
        finally:
            os.remove(chunk)  # Clean up the chunk file

    os.remove(audio_file_path)  # Clean up the original audio file
    return full_transcription

def generate_similar_text(transcription, openai_api_key):
    openai.api_key = openai_api_key
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" or another available model
            messages=[
                {"role": "system", "content": "I want to generate similar text."},
                {"role": "user", "content": transcription}
            ],
            max_tokens=150,
            temperature=0.7,
            n=3
        )
    except Exception as e:
        print(f"Error generating text: {e}")

    return [choice.message.content for choice in response.choices]

def generate_avatar(transcription, avatar_id, voice_id, heygen_key):
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
                "avatar_id": avatar_id,
                "scale": 1,
                "avatar_style": "normal",
                "offset": {
                  "x": 0,
                  "y": 0
                }
            },
            "voice": {
                "type": "text",
                "voice_id": voice_id,
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
    print(transcription)
    print(response.text)
    return response.json()

def get_avatar_list(heygen_key):
    get_avatar_url = "https://api.heygen.com/v2/avatars"
    headers = {"accept": "application/json", "x-api-key": heygen_key}
    response = requests.get(get_avatar_url, headers=headers)
    return response.json()

def get_voice_list(heygen_key):
    get_voice_url = "https://api.heygen.com/v2/voices"
    headers = {"accept": "application/json", "x-api-key": heygen_key}
    response = requests.get(get_voice_url, headers=headers)
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

# def download_video(video_url, output_filename):
#     print(video_url)
#     response = requests.get(video_url)
    
#     if response.status_code == 200:
#         with open(output_filename, 'wb') as video_file:
#             video_file.write(response.content)
#         print(f"Video downloaded successfully as {output_filename}.")
#     else:
#         raise Exception(f"Failed to download video. Status: {response.status_code}, Response: {response.text}")

def download_video(video_url, output_filename):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.instagram.com/"
    }
    
    response = requests.get(video_url, headers=headers, stream=True)
    
    if response.status_code == 200:
        with open(output_filename, 'wb') as video_file:
            for chunk in response.iter_content(chunk_size=1024):
                video_file.write(chunk)
        print(f"Video downloaded successfully as {output_filename}.")
    else:
        raise Exception(f"Failed to download video. Status: {response.status_code}, Response: {response.text}")

def extract_audio(video_path, audio_output_path="audio.mp3"):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_output_path
    ]
    subprocess.run(command)
    print(f"Audio extracted successfully: {audio_output_path}")

def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # You can use "small", "medium", or "large" models for better accuracy
    result = model.transcribe(audio_path)
    return result["text"]

# A dictionary to track the status of video generation tasks
tasks = {}

# Background task function to generate the video
def generate_video_task(task_id: str, text: str, avatar_id: str, voice_id: str):
    try:
        # Perform the long-running video generation
        response = generate_avatar(text, avatar_id, voice_id, heygen_key)
        print(f"response: {response}")
        video_id = response.get("data").get("video_id")
        print(f"Video ID: {video_id}")
        video_url = check_video_status(heygen_key, video_id)

        # Update the task status and video URL
        tasks[task_id] = {"status": "completed", "video_url": video_url}
    except Exception as e:
        # Handle errors and update the task status
        tasks[task_id] = {"status": "failed", "error": str(e)}

def format_duration(seconds):
    """Convert duration in seconds to '0:00:47' format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def get_instagram_videos(username: str):
    cl = Client(hiker_api_key)
    user = cl.user_by_username_v1(username)
    user_id = user.get("pk")
    videos = cl.user_clips_v1(user_id, 25)
    result = []
    for item in videos:
        video_id = item.get("id")
        video_title = item.get("caption_text", "")
        video_thumbnail = item.get("thumbnail_url")
        video_url = item.get("video_url")
        video_duration = item.get("video_duration", 0)
        likes = item.get("like_count", 0)
        views = item.get("play_count", 0)

        # Format the duration
        formatted_duration = format_duration(video_duration)

        result.append({
            'video_id': video_id,
            'title': video_title,
            'thumbnail': video_thumbnail,
            'url': video_url,
            'duration': formatted_duration,
            'likes': likes,
            'views': views
        })

    sorted_videos = sorted(result, key=lambda x: (x["likes"], x["views"]), reverse=True)

    return sorted_videos


def get_instagram_short_videos(username: str):
    cl = Client(hiker_api_key)
    user = cl.user_by_username_v1(username)
    user_id = user.get("pk")
    videos = cl.user_clips_v1(user_id, 25)
    result = []
    for item in videos:
        video_id = item.get("id")
        video_title = item.get("caption_text", "")
        video_thumbnail = item.get("thumbnail_url")
        video_url = item.get("video_url")
        video_duration = item.get("video_duration", 0)
        likes = item.get("like_count", 0)
        views = item.get("play_count", 0)
        if video_duration < 60:
            # Format the duration
            formatted_duration = format_duration(video_duration)

            result.append({
                'video_id': video_id,
                'title': video_title,
                'thumbnail': video_thumbnail,
                'url': video_url,
                'duration': formatted_duration,
                'likes': likes,
                'views': views
            })

    sorted_videos = sorted(result, key=lambda x: (x["likes"], x["views"]), reverse=True)

    return sorted_videos

@app.get("/videos")
def get_videos(profile_url: str, offset: int, limit: int):
    all_videos = get_sorted_videos(api_key, profile_url)
    paginated_videos = all_videos[offset : offset + limit]
    return {
        "videos": paginated_videos,
        "next_offset": offset + limit if offset + limit < len(all_videos) else None,
    }

@app.get("/short_videos")
def get_short_videos_endpoint(profile_url: str, offset: int = 0, limit: int = 10):
    all_short_videos = get_short_videos(api_key, profile_url)
    paginated_videos = all_short_videos[offset : offset + limit]
    return {
        "videos": paginated_videos,
        "next_offset": offset + limit if offset + limit < len(all_short_videos) else None,
    }

@app.get("/transcript_video")
def get_video_transcript(video_id: str):
    # Detect the video's language using the YouTube Data API
    detected_language = get_video_language(api_key, video_id)

    print(f"Language: {detected_language}")
    
    # Map ISO 639-1 language codes to Google Speech-to-Text language codes
    google_language_codes = {
        "en": "en-US",  # English
        "uk": "uk-UA",  # Ukrainian
        "ru": "ru-RU",  # Russian
        "es": "es-ES",  # Spanish
        "de": "de-DE",  # German
        "fr": "fr-FR"   # French
    }
    language_code = google_language_codes.get(detected_language, "en-US")
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    transcription = process_video(youtube_url, language_code)
    return transcription

@app.get("/generate_text")
def generate_text(transcription: str):
    similar_text = generate_similar_text(transcription, openai_api_key)
    return similar_text

@app.post("/generate_video")
def generate_video(request: GenerateVideoRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid4())
    tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(generate_video_task, task_id, request.text, request.avatar_id, request.voice_id)
    return {"task_id": task_id}

@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return {"status": "invalid", "error": "Task ID not found"}
    return task

@app.get("/avatar_list")
def fetch_avatar_list():
    response = get_avatar_list(heygen_key)
    return response.get("data").get("avatars")

@app.get("/voice_list")
def fetch_voice_list():
    response = get_voice_list(heygen_key)
    return response.get("data").get("voices")

@app.get("/insta_videos")
def fetch_insta_videos(username: str, offset: int, limit: int):
    all_videos = get_instagram_videos(username)
    paginated_videos = all_videos[offset : offset + limit]
    return {
        "videos": paginated_videos,
        "next_offset": offset + limit if offset + limit < len(all_videos) else None,
    }

@app.get("/insta_short_videos")
def fetch_insta_short_videos(username: str, offset: int, limit: int):
    all_videos = get_instagram_short_videos(username)
    paginated_videos = all_videos[offset : offset + limit]
    return {
        "videos": paginated_videos,
        "next_offset": offset + limit if offset + limit < len(all_videos) else None,
    }

@app.get("/proxy-image")
async def proxy_image(url: str = Query(...)):
    """
    Proxy endpoint to fetch an image from a given URL and serve it to the frontend.
    """
    try:
        # Fetch the image from the provided URL
        async with httpx.AsyncClient() as client:
            response = await client.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Return the image as a streaming response
            return StreamingResponse(
                response.iter_bytes(),  # Stream the image bytes
                media_type=response.headers['Content-Type']  # Ensure the correct content type
            )
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch image")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image: {str(e)}")

def download_instagram_video(url, output_format="wav"):
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
        ydl.download([url])
    return f"audio.{output_format}"

@app.get("/insta_transcript")
async def insta_transcript(url: str):
    audio_file_path = download_instagram_video(url, output_format="wav")
    audio_chunks = split_audio(audio_file_path)

    full_transcription = ''
    for chunk in audio_chunks:
        try:
            transcription = transcribe_audio_whisper(chunk, openai_api_key)
            print(transcription)
            full_transcription += transcription
        except Exception as e:
            print(f"Error transcribing chunk {chunk}: {e}")
        finally:
            time.sleep(1)  # Add a delay before removing the file
            os.remove(chunk)  # Clean up the chunk file
    os.remove(audio_file_path)
    
    return full_transcription


