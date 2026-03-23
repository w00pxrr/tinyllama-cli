/**
 * TinyLlama - AI for Node.js
 * 
 * Export types and utilities for use in other packages
 */

export * from './types.js';
export { TinyLlamaClient, getClient } from './client.js';

// React components (for use in React apps)
// Note: Import these separately: import { ChatBox } from 'tinyllama/react';
export { ChatBox, useChat } from './components/ChatBox.js';
export type { ChatBoxProps, UseChatReturn } from './types.js';