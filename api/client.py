"""
Python Client for Vercel AI API
Lets you make requests to the AI server from Python applications
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional

class AIApiClient:
    """Client for making AI inference requests"""
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        default_config: Dict[str, Any] = None
    ):
        """
        Initialize the AI API client
        
        Args:
            base_url: Base URL of the API (defaults to environment var AI_API_URL)
            api_key: API key for authentication (defaults to environment var AI_API_KEY)
            default_config: Default generation config to use
        """
        self.base_url = base_url or os.environ.get('AI_API_URL', 'http://localhost:3000')
        self.api_key = api_key or os.environ.get('AI_API_KEY', '')
        self.default_config = default_config or {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_new_tokens': 256,
            'repetition_penalty': 1.0
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Send a chat request to the AI API
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Optional generation config override
        
        Returns:
            Response dict with 'message' and 'model' keys
        """
        url = f"{self.base_url}/api/chat"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        payload = {
            'messages': messages,
            'config': config or self.default_config
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        return response.json()
    
    def chat_streaming(
        self,
        messages: List[Dict[str, str]],
        config: Dict[str, Any] = None,
        on_token: callable = None
    ) -> str:
        """
        Send a streaming chat request to the AI API
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Optional generation config override
            on_token: Callback function for each token received
        
        Returns:
            Full response text
        """
        url = f"{self.base_url}/api/chat/stream"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        payload = {
            'messages': messages,
            'config': config or self.default_config
        }
        
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        full_text = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    
                    token_data = json.loads(data)
                    if 'token' in token_data:
                        token = token_data['token']
                        full_text += token
                        if on_token:
                            on_token(token)
        
        return full_text
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the API
        
        Returns:
            Health status dict
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()


# Convenience function for simple usage
def chat(
    message: str,
    base_url: str = None,
    system_prompt: str = "You are a helpful AI assistant.",
    **generation_config
) -> str:
    """
    Simple chat function for quick usage
    
    Args:
        message: User message
        base_url: API base URL
        system_prompt: System prompt to use
        **generation_config: Generation parameters
    
    Returns:
        AI response text
    """
    client = AIApiClient(base_url=base_url)
    
    config = {
        'system_prompt': system_prompt,
        **generation_config
    }
    
    response = client.chat(
        messages=[{'role': 'user', 'content': message}],
        config=config
    )
    
    return response['message']['content']


# Example usage
if __name__ == '__main__':
    # Example: Simple chat
    print("=== Simple Chat Example ===")
    response = chat("Hello! How are you?")
    print(f"AI: {response}\n")
    
    # Example: Using client directly
    print("=== Client Example ===")
    client = AIApiClient()
    
    messages = [
        {'role': 'user', 'content': 'What is Python?'},
        {'role': 'assistant', 'content': 'Python is a programming language.'},
        {'role': 'user', 'content': 'What can I use it for?'}
    ]
    
    response = client.chat(messages, {'max_new_tokens': 128})
    print(f"AI: {response['message']['content']}\n")
    
    # Check health
    print("=== Health Check ===")
    health = client.health_check()
    print(f"Status: {health}")