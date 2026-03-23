"""
Vercel AI API Server
Downloads models on startup and provides AI inference API
"""

import os
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = os.environ.get('MODEL_ID', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/tmp/models')

# Global model and tokenizer
tokenizer = None
model = None

def download_model():
    """Download and cache the model on server startup"""
    global tokenizer, model
    
    logger.info(f"Downloading model: {MODEL_ID}")
    logger.info(f"Cache directory: {MODEL_CACHE_DIR}")
    
    # Create cache directory if needed
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR
        )
        logger.info("Tokenizer downloaded!")
        
        # Download model (using quantized version for performance)
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        logger.info("Model downloaded!")
        
        # Set model to eval mode
        model.eval()
        
        logger.info("Model ready for inference!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

class AIRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for AI inference"""
    
    def do_GET(self):
        """Health check endpoint"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            health = {
                'status': 'ok',
                'model': MODEL_ID,
                'model_loaded': model is not None,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            self.wfile.write(json.dumps(health).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """AI inference endpoint"""
        if self.path != '/api/chat':
            self.send_response(404)
            self.end_headers()
            return
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            request = json.loads(body.decode())
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode())
            return
        
        messages = request.get('messages', [])
        config = request.get('config', {})
        
        # Generate response
        try:
            response = generate_response(messages, config)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")

def generate_response(messages, config):
    """Generate AI response from messages"""
    global tokenizer, model
    
    if model is None or tokenizer is None:
        raise Exception("Model not loaded")
    
    # Build prompt from messages
    system_prompt = config.get('system_prompt', 'You are a helpful AI assistant.')
    prompt = f"<|system|>\n{system_prompt}\n"
    
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'user':
            prompt += f"<|user|>\n{content}\n"
        elif role == 'assistant':
            prompt += f"<|assistant|>\n{content}\n"
    
    prompt += "<|assistant|>\n"
    
    # Generation parameters
    temperature = config.get('temperature', 0.7)
    top_p = config.get('top_p', 0.9)
    top_k = config.get('top_k', 40)
    max_new_tokens = config.get('max_new_tokens', 256)
    repetition_penalty = config.get('repetition_penalty', 1.0)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0
        )
    
    # Decode output
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = generated[len(prompt):].strip()
    
    return {
        'message': {
            'role': 'assistant',
            'content': response_text
        },
        'model': MODEL_ID
    }

def start_server(port=3000):
    """Start the HTTP server"""
    server = HTTPServer(('0.0.0.0', port), AIRequestHandler)
    logger.info(f"Server started on port {port}")
    server.serve_forever()

def main():
    """Main entry point - download model then start server"""
    logger.info("=" * 50)
    logger.info("Vercel AI API Server")
    logger.info("=" * 50)
    
    # Download model
    success = download_model()
    
    if not success:
        logger.error("Failed to download model, starting anyway...")
    
    # Get port from environment (Vercel sets PORT)
    port = int(os.environ.get('PORT', 3000))
    
    # Start server
    start_server(port)

if __name__ == '__main__':
    main()