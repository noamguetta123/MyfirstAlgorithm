"""
Veo 3 Video Generator - Natural Body Movements
Uses Google Gen AI SDK to generate a video of an AI character
making natural body movements in front of a camera.
"""

import os
import time
import requests
from google import genai
from google.genai import types


API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

PROMPT = (
    "A beautiful young woman with long dark hair, wearing a brown halter dress "
    "and a butterfly necklace, standing in front of a white wall, making natural "
    "subtle body movements: gently tucking her hair behind her ear, slightly "
    "tilting her head, adjusting her hair with her hand, soft natural breathing "
    "movement, relaxed and candid — like a real person standing in front of a mirror. "
    "Cinematic, realistic, soft natural lighting, vertical portrait 9:16."
)


def generate_video(prompt: str, output_path: str = "output_video.mp4") -> str:
    client = genai.Client(api_key=API_KEY)

    print("Sending request to Veo 3...")
    operation = client.models.generate_videos(
        model="veo-3.0-generate-preview",
        prompt=prompt,
        config=types.GenerateVideoConfig(
            aspect_ratio="9:16",
            duration_seconds=8,
            number_of_videos=1,
            enhance_prompt=True,
        ),
    )

    print("Waiting for video generation (this may take 2-3 minutes)...")
    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)
        print("  Still processing...")

    if operation.response and operation.response.generated_videos:
        video = operation.response.generated_videos[0]
        video_bytes = client.files.download(file=video.video)

        with open(output_path, "wb") as f:
            f.write(video_bytes)

        print(f"Video saved to: {output_path}")
        return output_path
    else:
        raise RuntimeError(f"Video generation failed: {operation}")


if __name__ == "__main__":
    try:
        result = generate_video(PROMPT, "natural_movements.mp4")
        print(f"Done! Video saved: {result}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set GEMINI_API_KEY environment variable")
        print("  2. Installed: pip install google-genai")
        print("  3. Veo 3 access enabled on your Google AI account")
