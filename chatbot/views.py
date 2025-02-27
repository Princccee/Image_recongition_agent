from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from huggingface_hub import InferenceClient
import requests
from dotenv import load_dotenv
from PIL import Image
import os
import json

load_dotenv()

# Get API key and model name from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Fireworks AI Provider
client = InferenceClient(provider="fireworks-ai", api_key=HF_API_KEY)

@api_view(["POST"])
def process_image(request):
    """
    API endpoint to receive an image and user query, then send it to Hugging Face LLaMA model.
    """
    if "image" not in request.FILES or "query" not in request.data:
        return Response({"error": "Both image and query are required"}, status=400)

    # Save image temporarily
    image_file = request.FILES["image"]
    file_name = default_storage.save(image_file.name, ContentFile(image_file.read()))
    image_url = request.build_absolute_uri(default_storage.url(file_name))

    # User's query
    user_query = request.data["query"]

    # Construct LLaMA request
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {"type": "image_url", "image_url": {"url": image_url}}
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
