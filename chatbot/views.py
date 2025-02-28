from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from huggingface_hub import InferenceClient
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError
import io
import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

client = InferenceClient(provider="fireworks-ai", api_key=HF_API_KEY)

def upload_to_google_drive(image_file):
    """Uploads an in-memory image file directly to Google Drive and returns a public URL"""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=credentials)

    file_metadata = {
        "name": image_file.name,
        "parents": [GOOGLE_DRIVE_FOLDER_ID],  # Upload to specific folder
    }

    media = MediaIoBaseUpload(image_file, mimetype=image_file.content_type)

    try:
        file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        file_id = file.get("id")

        # Set public permissions
        service.permissions().create(
            fileId=file_id,
            body={"role": "reader", "type": "anyone"},
        ).execute()

        return f"https://drive.google.com/uc?id={file_id}"
    except HttpError as e:
        print(f"Upload failed: {e}")
        return None


@api_view(["POST"])
def process_image(request):
    """Process an image and a query using the model"""
    
    if "image" not in request.FILES or "query" not in request.data:
        return Response({"error": "Both image and query are required"}, status=400)
    
    image_file = request.FILES["image"]
    user_query = request.data["query"]

    # Debug: Check if the file is valid
    print(f"Received file: {image_file.name}, Size: {image_file.size} bytes, Type: {image_file.content_type}")

    if image_file.size == 0:
        return Response({"error": "Empty image file received"}, status=400)
    
    # Validate image format
    try:
        image = Image.open(image_file)
        image.verify()  # Check if it's a valid image
    except Exception as e:
        return Response({"error": f"Invalid image file: {str(e)}"}, status=400)

    # Directly upload to Google Drive
    image_url = upload_to_google_drive(image_file)
    if not image_url:
        return Response({"error": "Failed to upload image"}, status=500)

    print(f"Image uploaded successfully: {image_url}")

    # Prepare the payload for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        }
    ]

    # Call the model
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages, 
            max_tokens=500
        )
        response_text = completion.choices[0].message.content
        print(response_text)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

    return Response({"response": response_text})
