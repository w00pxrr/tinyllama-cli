/**
 * TinyLlama Client - Wraps Transformers.js for local inference
 */

import type { GenerationConfig, ChatMessage } from './types.js';

// Dynamic import for Transformers.js (works in Node and Browser)
let pipeline: any = null;

async function getPipeline() {
  if (!pipeline) {
    const { pipeline: tfPipeline } = await import('@xenova/transformers');
    pipeline = tfPipeline;
  }
  return pipeline;
}

export class TinyLlamaClient {
  public model: any = null; // Public for server API access
  public modelId: string;
  private device: string;
  
  constructor(modelId: string = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', device: string = 'cpu') {
    this.modelId = modelId;
    this.device = device;
  }
  
  async load(): Promise<void> {
    const pipelineFn = await getPipeline();
    console.log(`Loading model: ${this.modelId}`);
    
    this.model = await pipelineFn('text-generation', this.modelId, {
      dtype: 'q4', // Quantized for performance
    });
    
    console.log('Model loaded!');
  }
  
  async generate(
    messages: ChatMessage[],
    systemPrompt: string = 'You are a helpful AI assistant.',
    config: GenerationConfig
  ): Promise<string> {
    if (!this.model) {
      await this.load();
    }
    
    // Build prompt from messages
    let prompt = `<|system|>\n${systemPrompt}\n`;
    
    for (const msg of messages) {
      if (msg.role === 'user') {
        prompt += `<|user|>\n${msg.content}\n`;
      } else if (msg.role === 'assistant') {
        prompt += `<|assistant|>\n${msg.content}\n`;
      }
    }
    
    prompt += `<|assistant|>\n`;
    
    const output = await this.model(prompt, {
      temperature: config.temperature,
      top_p: config.top_p,
      top_k: config.top_k,
      repetition_penalty: config.repetition_penalty,
      max_new_tokens: config.max_new_tokens,
      do_sample: config.do_sample,
    });
    
    const generated = output[0].generated_text;
    return generated.slice(prompt.length).trim();
  }
  
  isLoaded(): boolean {
    return this.model !== null;
  }
  
  async unload(): Promise<void> {
    this.model = null;
  }
}

// Singleton instance for React hooks
let globalClient: TinyLlamaClient | null = null;

export function getClient(modelId?: string): TinyLlamaClient {
  if (!globalClient || (modelId && globalClient.modelId !== modelId)) {
    globalClient = new TinyLlamaClient(modelId);
  }
  return globalClient;
}