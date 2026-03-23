# TinyLlama

TypeScript package for TinyLlama AI - includes CLI and React components.

## Installation

```bash
npm install tinyllama
# or
npm install .
```

## Usage

### CLI

```bash
npm start
# or
npx tinyllama
```

Commands:
- `/help` - Show help
- `/clear` - Clear chat
- `/save` - Save transcript
- `/exit` - Quit
- `/models` - List models

### React Component

```tsx
import { ChatBox } from 'tinyllama/react';

function App() {
  return (
    <ChatBox 
      model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
      darkMode={false}
      onModelChange={(modelId) => console.log(modelId)}
    />
  );
}
```

### Use Chat Hook

```tsx
import { useChat } from 'tinyllama/react';

function MyComponent() {
  const { messages, isLoading, sendMessage, clearHistory } = useChat();
  
  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i}>{msg.content}</div>
      ))}
      <button onClick={() => sendMessage('Hello!')}>Send</button>
    </div>
  );
}
```

## Build

```bash
npm run build
```

This produces:
- `dist/index.js` - Main package
- `dist/components/ChatBox.js` - React components
- Type definitions included