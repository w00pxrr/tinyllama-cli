/**
 * Next.js API Route handler for server-side AI inference
 * 
 * Usage in Next.js App Router (App Router):
 * 1. Create file: app/api/chat/route.ts
 * 2. Import and export this handler
 * 
 * Example:
 * ```ts
 * import { POST } from 'tinyllama/server'
 * export { POST }
 * ```
 */

import { getClient } from '../client.js'
import type { GenerationConfig, ChatMessage } from '../types.js'

export interface ChatRequest {
  messages: ChatMessage[]
  model?: 'tinyllama' | 'smollm' | 'qwen'
  systemPrompt?: string
  config?: Partial<GenerationConfig>
}

export interface ChatResponse {
  message: ChatMessage
  done: boolean
}

// Map model names to HuggingFace model IDs
const MODEL_MAP: Record<string, string> = {
  tinyllama: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  smollm: 'HuggingFaceTB/SmolLM2-135M',
  qwen: 'Qwen/Qwen2.5-0.5B-Instruct',
}

const DEFAULT_CONFIG: GenerationConfig = {
  temperature: 0.65,
  top_p: 0.9,
  top_k: 40,
  repetition_penalty: 1.1,
  max_new_tokens: 256,
  do_sample: true,
}

export async function POST(request: Request): Promise<Response> {
  try {
    const body: ChatRequest = await request.json()
    
    if (!body.messages || !Array.isArray(body.messages)) {
      return Response.json(
        { error: 'messages array is required' },
        { status: 400 }
      )
    }

    const modelId = MODEL_MAP[body.model || 'tinyllama']
    const config = { ...DEFAULT_CONFIG, ...body.config }
    const systemPrompt = body.systemPrompt || 'You are a helpful AI assistant.'

    const client = getClient(modelId)
    await client.load()

    const response = await client.generate(body.messages, systemPrompt, config)

    const chatResponse: ChatResponse = {
      message: {
        role: 'assistant',
        content: response
      },
      done: true
    }

    return Response.json(chatResponse)
  } catch (error) {
    console.error('AI API Error:', error)
    return Response.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

/**
 * Create a Next.js API route handler for streaming responses
 * 
 * Usage:
 * ```ts
 * import { createStreamingHandler } from 'tinyllama/server'
 * export const POST = createStreamingHandler()
 * ```
 */
export function createStreamingHandler() {
  return async function handler(request: Request): Promise<Response> {
    try {
      const body: ChatRequest = await request.json()
      
      if (!body.messages || !Array.isArray(body.messages)) {
        return Response.json(
          { error: 'messages array is required' },
          { status: 400 }
        )
      }

      const modelId = MODEL_MAP[body.model || 'tinyllama']
      const config: GenerationConfig = { ...DEFAULT_CONFIG, ...body.config }
      const systemPrompt = body.systemPrompt || 'You are a helpful AI assistant.'

      const encoder = new TextEncoder()
      const stream = new ReadableStream({
        async start(controller) {
          try {
            const client = getClient(modelId)
            await client.load()
            
            // Build prompt from messages
            let prompt = `<|system|>\\n${systemPrompt}\\n`
            
            for (const msg of body.messages) {
              if (msg.role === 'user') {
                prompt += `<|user|>\\n${msg.content}\\n`
              } else if (msg.role === 'assistant') {
                prompt += `<|assistant|>\\n${msg.content}\\n`
              }
            }
            
            prompt += `<|assistant|>\\n`
            
            // Note: Streaming callback not supported in current client
            // For true streaming, you'd need to modify the client
            const output = await client.model(prompt, {
              temperature: config.temperature,
              top_p: config.top_p,
              top_k: config.top_k,
              repetition_penalty: config.repetition_penalty,
              max_new_tokens: config.max_new_tokens,
              do_sample: config.do_sample,
            })
            
            const generated = output[0].generated_text
            const content = generated.slice(prompt.length).trim()
            
            // Send the complete response
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content })}\\n\\n`))
            controller.enqueue(encoder.encode('data: {"done": true}\\n\\n'))
            controller.close()
          } catch (err) {
            controller.error(err)
          }
        }
      })

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        }
      })
    } catch (error) {
      return Response.json(
        { error: error instanceof Error ? error.message : 'Internal server error' },
        { status: 500 }
      )
    }
  }
}