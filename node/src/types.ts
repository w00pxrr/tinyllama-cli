/**
 * TypeScript types for TinyLlama
 */

export interface GenerationConfig {
  temperature: number;
  top_p: number;
  top_k: number;
  repetition_penalty: number;
  max_new_tokens: number;
  do_sample: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
}

export const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    name: 'TinyLlama',
    description: '1.1B parameters - Fast and lightweight',
  },
  {
    id: 'HuggingFaceTB/SmolLM2-135M',
    name: 'SmolLM2',
    description: '135M parameters - Ultra fast',
  },
  {
    id: 'Qwen/Qwen2.5-0.5B-Instruct',
    name: 'Qwen',
    description: '500M parameters - Good balance',
  },
];

export const DEFAULT_CONFIG: GenerationConfig = {
  temperature: 0.65,
  top_p: 0.9,
  top_k: 40,
  repetition_penalty: 1.1,
  max_new_tokens: 256,
  do_sample: true,
};

export interface ChatBoxProps {
  model?: string;
  systemPrompt?: string;
  config?: Partial<GenerationConfig>;
  className?: string;
  onModelChange?: (modelId: string) => void;
  darkMode?: boolean;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (content: string) => Promise<void>;
  clearHistory: () => void;
}