# Car Enhancer Backend

FastAPI backend for enhancing car photos for marketplace/listing use.

The service applies:
- a deterministic global pre-grade (color/contrast/brightness/sharpness), then
- an OpenAI image edit pass (`gpt-image-1.5`) using strict preservation instructions.

It preserves original composition by letterboxing to a supported API canvas, then crops/resizes back to original dimensions.

## Features

- `POST /enhance` endpoint for image enhancement (`multipart/form-data`)
- `POST /` alias for the same enhancement endpoint
- `GET /` built-in HTML upload/demo page
- `GET /health` health check endpoint
- Containerized runtime via Docker + Docker Compose
- Vercel-compatible Python serverless routing (`vercel.json`)

## Tech Stack

- FastAPI
- Pillow + NumPy
- OpenAI Python SDK (`openai>=1.30.0`)
- Uvicorn
- Python 3.11

## Project Structure

```
.
├─ api/
│  ├─ index.py                  # FastAPI app + routes + OpenAI image edit flow
│  └─ utils/
│     └─ opencv_pipeline.py     # Deterministic global pre-grade
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
└─ vercel.json
```

## Environment Variables

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=your_openai_api_key
```

Required for image enhancement endpoints.

## Run Locally (Python)

1. Create/activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API:

```bash
uvicorn api.index:app --host 0.0.0.0 --port 8000
```

4. Open:

- App page: `http://localhost:8000/`
- Health: `http://localhost:8000/health`

## Run with Docker Compose

```bash
docker compose up -d --build
```

Service is exposed on `http://localhost:8000`.

Stop:

```bash
docker compose down
```

## API

### `GET /`

Returns a simple HTML demo page for uploading an image and viewing the result.

### `GET /health`

Returns:

```json
{ "status": "ok" }
```

### `POST /enhance`

Also available as `POST /`.

#### Request

- Content-Type: `multipart/form-data`
- Fields:
	- `file` (required): image file
	- `strength` (optional): float in `0.0..1.0` (default `0.35`)

#### Example (`curl`)

```bash
curl -X POST "http://localhost:8000/enhance" \
	-F "file=@car.jpg" \
	-F "strength=0.35" \
	--output enhanced.jpg
```

#### Response

- Success: binary image (`image/jpeg` or `image/png`) rendered inline
- Error: JSON with `error` and `trace` (HTTP 500)

Additional response headers include:
- `X-AI-USED`
- `X-ORIG`
- `X-API-CANVAS`
- `X-BOX`
- `X-STRENGTH`

## Processing Pipeline (High Level)

1. Read uploaded image and normalize orientation (`EXIF` transpose).
2. Choose nearest supported OpenAI canvas size (`1024x1024`, `1536x1024`, or `1024x1536`) by aspect ratio.
3. Letterbox original image to that canvas (no crop, no distortion).
4. Apply deterministic global pre-grade (`strength` controls intensity).
5. Send both pre-graded and original canvas images to OpenAI `images.edit` for anchored enhancement.
6. Crop back to original content region and resize to original resolution.
7. Return final image in original-friendly format.

## CORS

Configured to allow HTTPS origins via regex:

- `^https://.*$`

## Deploying to Vercel

This repo already includes `vercel.json` mapping all routes to `api/index.py`.

Typical steps:

1. Import the repository into Vercel.
2. Set environment variable `OPENAI_API_KEY` in project settings.
3. Deploy.

After deploy, visiting `/` opens the upload UI and `POST /enhance` serves the API.

## Troubleshooting

- `OPENAI_API_KEY is not set`
	- Ensure `.env` exists locally and includes a valid key.
	- For Docker, verify Compose is loading `.env` and variable is present.
	- For Vercel, set env var in project settings.

- Enhancement endpoint returns 500
	- Check server logs for full traceback.
	- Verify API key validity and model access for `gpt-image-1.5`.
	- Confirm uploaded file is a valid image.

- Browser cannot call API from another domain
	- Current CORS policy only allows HTTPS origins.

## License

No license file is currently defined in this repository.