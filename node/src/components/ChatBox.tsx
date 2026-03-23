/**
 * ChatBox - React component for TinyLlama AI chat
 * Import this into your React app to add AI chat functionality
 */

import React, { useState, useRef, useEffect } from 'react';
import { AVAILABLE_MODELS, DEFAULT_CONFIG, type ChatMessage, type GenerationConfig, type ChatBoxProps } from '../types.js';
import { getClient } from '../client.js';

// Simple CSS-in-JS styles (or import your own CSS)
const styles = {
  container: {
    fontFamily: 'system-ui, -apple-system, sans-serif',
    maxWidth: '600px',
    margin: '0 auto',
    padding: '20px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
    padding: '15px',
    borderRadius: '8px',
  },
  modelSelect: {
    padding: '8px 12px',
    borderRadius: '6px',
    border: '1px solid #ccc',
    fontSize: '14px',
    cursor: 'pointer',
  },
  messages: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '12px',
    maxHeight: '400px',
    overflowY: 'auto' as const,
    marginBottom: '20px',
    padding: '10px',
  },
  message: {
    padding: '12px 16px',
    borderRadius: '12px',
    maxWidth: '80%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007bff',
    color: 'white',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#f0f0f0',
    color: '#333',
  },
  inputContainer: {
    display: 'flex',
    gap: '10px',
  },
  input: {
    flex: 1,
    padding: '12px',
    borderRadius: '8px',
    border: '1px solid #ccc',
    fontSize: '14px',
    fontFamily: 'inherit',
  },
  sendButton: {
    padding: '12px 24px',
    borderRadius: '8px',
    border: 'none',
    backgroundColor: '#007bff',
    color: 'white',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
  },
  loading: {
    color: '#666',
    fontStyle: 'italic',
  },
  error: {
    color: '#dc3545',
    padding: '10px',
    backgroundColor: '#f8d7da',
    borderRadius: '6px',
    marginBottom: '10px',
  },
};

// Dark mode styles
const darkStyles = {
  ...styles,
  header: {
    ...styles.header,
    backgroundColor: '#1a1a1a',
  },
  message: {
    ...styles.assistantMessage,
    backgroundColor: '#333',
    color: '#fff',
  },
  input: {
    ...styles.input,
    backgroundColor: '#333',
    borderColor: '#555',
    color: 'white',
  },
};

export function ChatBox({
  model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  systemPrompt = 'You are a helpful AI assistant.',
  config = DEFAULT_CONFIG,
  className = '',
  onModelChange,
  darkMode = false,
}: ChatBoxProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState(model);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeStyles = darkMode ? darkStyles : styles;

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;
    setCurrentModel(newModel);
    onModelChange?.(newModel);
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      const client = getClient(currentModel);
      
      // Load model if not loaded
      if (!client.isLoaded()) {
        await client.load();
      }

      const response = await client.generate(
        [...messages, userMessage],
        systemPrompt,
        { ...DEFAULT_CONFIG, ...config }
      );

      const assistantMessage: ChatMessage = { role: 'assistant', content: response };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate response');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearHistory = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className={className} style={activeStyles.container}>
      {/* Header with model selector */}
      <div style={activeStyles.header}>
        <select
          value={currentModel}
          onChange={handleModelChange}
          style={activeStyles.modelSelect}
        >
          {AVAILABLE_MODELS.map(m => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
        <button
          onClick={clearHistory}
          style={{
            padding: '8px 12px',
            borderRadius: '6px',
            border: '1px solid #ccc',
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          Clear
        </button>
      </div>

      {/* Error message */}
      {error && <div style={activeStyles.error}>{error}</div>}

      {/* Messages */}
      <div style={activeStyles.messages}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              ...activeStyles.message,
              ...(msg.role === 'user' ? activeStyles.userMessage : activeStyles.assistantMessage),
            }}
          >
            {msg.content}
          </div>
        ))}
        {isLoading && (
          <div style={activeStyles.loading}>Thinking...</div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={activeStyles.inputContainer}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={isLoading}
          style={activeStyles.input}
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          style={{
            ...activeStyles.sendButton,
            opacity: isLoading || !input.trim() ? 0.6 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

// React hook for programmatic chat control
export function useChat(
  model?: string,
  systemPrompt?: string,
  config?: Partial<GenerationConfig>
) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = async (content: string) => {
    const userMessage: ChatMessage = { role: 'user', content };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const client = getClient(model);
      if (!client.isLoaded()) {
        await client.load();
      }

      const response = await client.generate(
        [...messages, userMessage],
        systemPrompt || 'You are a helpful AI assistant.',
        { ...DEFAULT_CONFIG, ...config }
      );

      const assistantMessage: ChatMessage = { role: 'assistant', content: response };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate');
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = () => {
    setMessages([]);
    setError(null);
  };

  return { messages, isLoading, error, sendMessage, clearHistory };
}

export default ChatBox;