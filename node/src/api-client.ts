/**
 * Node.js Client for Vercel AI API
 * Lets you make requests to the AI server from Node.js applications
 */

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface GenerationConfig {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_new_tokens?: number;
  repetition_penalty?: number;
  system_prompt?: string;
}

export interface ChatResponse {
  message: ChatMessage;
  model: string;
}

export interface HealthStatus {
  status: 'ok' | 'error';
  model: string;
  model_loaded: boolean;
  device: string;
}

/**
 * Client for making AI inference requests to the Vercel API
 */
export class AIApiClient {
  private baseUrl: string;
  private apiKey: string;
  private defaultConfig: GenerationConfig;

  constructor(options: {
    baseUrl?: string;
    apiKey?: string;
    defaultConfig?: GenerationConfig;
  } = {}) {
    // Get env vars safely (works in Node.js and browser)
    const getEnv = (key: string, fallback: string): string => {
      if (typeof globalThis !== 'undefined' && (globalThis as any).process?.env) {
        return (globalThis as any).process.env[key] || fallback;
      }
      return fallback;
    };

    this.baseUrl = options.baseUrl || getEnv('AI_API_URL', 'http://localhost:3000');
    this.apiKey = options.apiKey || getEnv('AI_API_KEY', '');
    this.defaultConfig = options.defaultConfig || {
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
      max_new_tokens: 256,
      repetition_penalty: 1.0,
    };
  }

  /**
   * Send a chat request to the AI API
   */
  async chat(
    messages: ChatMessage[],
    config?: GenerationConfig
  ): Promise<ChatResponse> {
    const url = `${this.baseUrl}/api/chat`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const payload = {
      messages,
      config: config || this.defaultConfig,
    };

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`AI API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  /**
   * Send a streaming chat request to the AI API
   */
  async *chatStreaming(
    messages: ChatMessage[],
    config?: GenerationConfig
  ): AsyncGenerator<string> {
    const url = `${this.baseUrl}/api/chat/stream`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const payload = {
      messages,
      config: config || this.defaultConfig,
    };

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`AI API error: ${response.status} - ${error}`);
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);

          if (data === '[DONE]') {
            return;
          }

          try {
            const tokenData = JSON.parse(data);
            if (tokenData.token) {
              yield tokenData.token;
            }
          } catch {
            // Skip invalid JSON
          }
        }
      }
    }
  }

  /**
   * Check the health of the API
   */
  async healthCheck(): Promise<HealthStatus> {
    const url = `${this.baseUrl}/health`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }

    return response.json();
  }
}

/**
 * Simple chat function for quick usage
 */
export async function chat(
  message: string,
  options: {
    baseUrl?: string;
    apiKey?: string;
    systemPrompt?: string;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    max_new_tokens?: number;
    repetition_penalty?: number;
  } = {}
): Promise<string> {
  const client = new AIApiClient({
    baseUrl: options.baseUrl,
    apiKey: options.apiKey,
  });

  const config: GenerationConfig = {
    system_prompt: options.systemPrompt || 'You are a helpful AI assistant.',
    temperature: options.temperature,
    top_p: options.top_p,
    top_k: options.top_k,
    max_new_tokens: options.max_new_tokens,
    repetition_penalty: options.repetition_penalty,
  };

  const response = await client.chat(
    [{ role: 'user', content: message }],
    config
  );

  return response.message.content;
}

// Example usage
if (typeof globalThis !== 'undefined' && (globalThis as any).process?.argv) {
  const argv = (globalThis as any).process.argv;
  if (import.meta.url === `file://${argv[1]}`) {
  (async () => {
    console.log('=== Simple Chat Example ===');
    const response = await chat('Hello! How are you?');
    console.log(`AI: ${response}\n`);

    console.log('=== Client Example ===');
    const client = new AIApiClient();

    const messages: ChatMessage[] = [
      { role: 'user', content: 'What is TypeScript?' },
      { role: 'assistant', content: 'TypeScript is a typed superset of JavaScript.' },
      { role: 'user', content: 'What can I use it for?' },
    ];

    const chatResponse = await client.chat(messages, { max_new_tokens: 128 });
    console.log(`AI: ${chatResponse.message.content}\n`);

    console.log('=== Health Check ===');
    const health = await client.healthCheck();
    console.log('Status:', health);
  })();
}