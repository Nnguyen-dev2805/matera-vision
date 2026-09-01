import base64

import httpx

from matera.vlm.models import VlmRequest, VlmResponse


class GeminiVlmClient:
    def __init__(self, api_key: str, *, timeout_s: float = 60.0):
        self.api_key = api_key
        self.timeout_s = timeout_s

    def generate_json(self, request: VlmRequest) -> VlmResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent?key={self.api_key}"  # noqa: E501

        with open(request.image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": request.prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_data}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": request.temperature,
                "response_mime_type": "application/json",
            },
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(url, json=payload)

            # Allow raising HTTP errors if the request itself failed
            response.raise_for_status()

            data = response.json()

            try:
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                # Handle cases where the response structure is unexpected
                # We return the raw string of the data to help debugging
                raw_text = str(data)

            return VlmResponse(
                raw_text=raw_text,
                provider_metadata={
                    "usageMetadata": data.get("usageMetadata", {}),
                    "full_response": data,
                },
            )

    def classify_q14(self, request: VlmRequest) -> VlmResponse:
        return self.generate_json(request)
