from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from huggingface_hub import InferenceClient
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import requests
from dotenv import load_dotenv
from PIL import Image
import os
import json

load_dotenv()

# Get API key and model name from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

# Fireworks AI Provider
client = InferenceClient(provider="fireworks-ai", api_key=HF_API_KEY)

# Google Drive API
def upload_to_google_drive(image_path):
    """Uploads an image to Google Drive and returns a public URL"""
    
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    # SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")  # JSON file downloaded from Google Cloud

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    service = build("drive", "v3", credentials=credentials)

    file_metadata = {
        "name": os.path.basename(image_path),
        "parents": [GOOGLE_DRIVE_FOLDER_ID],  # Upload to specific folder
    }

    media = MediaFileUpload(image_path, mimetype="image/jpeg")
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    file_id = file.get("id")

    # Make file public
    service.permissions().create(
        fileId=file_id, body={"role": "reader", "type": "anyone"}
    ).execute()

    # Get public URL
    public_url = f"https://drive.google.com/uc?id={file_id}"
    return public_url


@api_view(["POST"])
def process_image(request):
    """Process an image and a query using the model"""
    
    if "image" not in request.FILES or "query" not in request.data:
        return Response({"error": "Both image and query are required"}, status=400)
    
    image_file = request.FILES["image"]
    
     # Validate image format
    try:
        image = Image.open(ContentFile(image_file.read()))
        image.verify()  # Check if it is a valid image
    except Exception as e:
        return Response({"error": f"Invalid image file: {str(e)}"}, status=400)

    # Save the uploaded image temporarily
    file_name = default_storage.save(image_file.name, ContentFile(image_file.read()))
    local_image_path = default_storage.path(file_name)

    # Upload image to Google Drive
    image_url = upload_to_google_drive(local_image_path)
    if not image_url:
        return Response({"error": "Failed to upload image"}, status=500)

    user_query = request.data["query"]
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
            model=MODEL_NAME, messages=messages, max_tokens=500
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        return Response({"error": str(e)}, status=500)

    return Response({"response": response_text})
