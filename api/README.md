# Vercel AI API Server

Python server that downloads AI models on startup and provides an API for inference.

## Quick Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

## Local Development

```bash
pip install -r requirements.txt
python main.py
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat` - Chat completion

## Environment Variables

- `MODEL_ID` - HuggingFace model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `MODEL_CACHE_DIR` - Model cache directory (default: /tmp/models)
- `PORT` - Server port (default: 3000)
- `AI_API_KEY` - Optional API key for authentication

## Python Client

```python
from client import AIApiClient, chat

# Simple usage
response = chat("Hello!")
print(response)

# Full client
client = AIApiClient(base_url="https://your-app.vercel.app")
result = client.chat([
    {"role": "user", "content": "Hi!"}
])
print(result["message"]["content"])
```

## Node.js Client

```bash
npm install tinyllama
```

```typescript
import { chat, AIApiClient } from 'tinyllama';

// Simple usage
const response = await chat("Hello!");

// Full client
const client = new AIApiClient({
  baseUrl: "https://your-app.vercel.app"
});

const result = await client.chat([
  { role: "user", content: "Hi!" }
]);
console.log(result.message.content);