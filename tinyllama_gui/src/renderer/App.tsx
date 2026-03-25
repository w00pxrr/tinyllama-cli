import React, { useState, useEffect, useRef } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Slider from '@radix-ui/react-slider';

declare global {
  interface Window {
    api: {
      sendMessage: (message: string) => Promise<{ response?: string; error?: string }>;
      getModes: () => Promise<{ modes: Array<{ id: string; name: string; description: string }> }>;
      setMode: (modeId: string) => Promise<{ success?: boolean; error?: string }>;
      getSettings: () => Promise<{ settings: { temperature: number; maxTokens: number; topP: number } }>;
      saveSettings: (settings: object) => Promise<{ success: boolean }>;
      feedback: (type: 'like' | 'dislike', message: string) => Promise<{ success: boolean }>;
      selectModel: () => Promise<string | null>;
      downloadModel: () => Promise<{ success?: boolean; message?: string; error?: string }>;
    };
  }
}

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentMode, setCurrentMode] = useState('chat');
  const [modes, setModes] = useState<Array<{ id: string; name: string; description: string }>>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({ temperature: 0.7, maxTokens: 512, topP: 0.9 });
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadModes();
    loadSettings();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadModes = async () => {
    try {
      const result = await window.api.getModes();
      setModes(result.modes);
    } catch (e) {
      console.error('Failed to load modes:', e);
    }
  };

  const loadSettings = async () => {
    try {
      const result = await window.api.getSettings();
      setSettings(result.settings);
    } catch (e) {
      console.error('Failed to load settings:', e);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: input
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const result = await window.api.sendMessage(input);
      if (result.response) {
        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          role: 'assistant',
          content: result.response
        }]);
      } else if (result.error) {
        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          role: 'assistant',
          content: `Error: ${result.error}`
        }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Failed to get response'
      }]);
    }
    setIsLoading(false);
  };

  const handleFeedback = async (type: 'like' | 'dislike', message: string) => {
    await window.api.feedback(type, message);
  };

  const handleModeChange = async (modeId: string) => {
    await window.api.setMode(modeId);
    setCurrentMode(modeId);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>TinyLlama CLI</h1>
        
        <div style={styles.headerButtons}>
          <button 
            style={styles.downloadButton} 
            onClick={async () => {
              try {
                await window.api.downloadModel();
              } catch (e) {
                console.error('Failed to download model:', e);
              }
            }}
          >
            Download Model
          </button>
          
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <button style={styles.modeButton}>
                Mode: {modes.find(m => m.id === currentMode)?.name || 'Chat'}
              </button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content style={styles.dropdownContent}>
                {modes.map(mode => (
                  <DropdownMenu.Item
                    key={mode.id}
                    onSelect={() => handleModeChange(mode.id)}
                    style={styles.dropdownItem}
                  >
                    <div style={styles.modeItemName}>{mode.name}</div>
                    <div style={styles.modeItemDesc}>{mode.description}</div>
                  </DropdownMenu.Item>
                ))}
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>

          <button style={styles.settingsButton} onClick={() => setSettingsOpen(true)}>
            Settings
          </button>
        </div>
      </header>

      <main style={styles.main}>
        <div style={styles.messages}>
          {messages.length === 0 && (
            <div style={styles.emptyState}>
              <p>Welcome to TinyLlama CLI!</p>
              <p style={styles.emptySubtext}>Send a message to start chatting</p>
            </div>
          )}
          {messages.map(msg => (
            <div key={msg.id} style={msg.role === 'user' ? styles.userMessage : styles.assistantMessage}>
              <div style={styles.messageContent}>
                {msg.content}
              </div>
              {msg.role === 'assistant' && (
                <div style={styles.feedbackButtons}>
                  <button 
                    style={styles.feedbackButton}
                    onClick={() => handleFeedback('like', msg.content)}
                    title="Like"
                  >
                    👍
                  </button>
                  <button 
                    style={styles.feedbackButton}
                    onClick={() => handleFeedback('dislike', msg.content)}
                    title="Dislike"
                  >
                    👎
                  </button>
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div style={styles.assistantMessage}>
              <div style={styles.messageContent}>
                <span style={styles.loadingDots}>Thinking...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer style={styles.footer}>
        <input
          style={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button 
          style={styles.sendButton} 
          onClick={sendMessage}
          disabled={isLoading || !input.trim()}
        >
          Send
        </button>
      </footer>

      <Dialog.Root open={settingsOpen} onOpenChange={setSettingsOpen}>
        <Dialog.Portal>
          <Dialog.Overlay style={styles.dialogOverlay} />
          <Dialog.Content style={styles.dialogContent}>
            <Dialog.Title style={styles.dialogTitle}>Settings</Dialog.Title>
            
            <div style={styles.settingGroup}>
              <label style={styles.settingLabel}>Temperature: {settings.temperature}</label>
              <Slider.Root 
                style={styles.slider} 
                value={[settings.temperature]} 
                onValueChange={([v]) => setSettings(s => ({ ...s, temperature: v }))}
                max={1}
                step={0.1}
              >
                <Slider.Track style={styles.sliderTrack}>
                  <Slider.Range style={styles.sliderRange} />
                </Slider.Track>
                <Slider.Thumb style={styles.sliderThumb} />
              </Slider.Root>
            </div>

            <div style={styles.settingGroup}>
              <label style={styles.settingLabel}>Max Tokens: {settings.maxTokens}</label>
              <Slider.Root 
                style={styles.slider} 
                value={[settings.maxTokens]} 
                onValueChange={([v]) => setSettings(s => ({ ...s, maxTokens: v }))}
                max={2048}
                step={64}
              >
                <Slider.Track style={styles.sliderTrack}>
                  <Slider.Range style={styles.sliderRange} />
                </Slider.Track>
                <Slider.Thumb style={styles.sliderThumb} />
              </Slider.Root>
            </div>

            <div style={styles.settingGroup}>
              <label style={styles.settingLabel}>Top P: {settings.topP}</label>
              <Slider.Root 
                style={styles.slider} 
                value={[settings.topP]} 
                onValueChange={([v]) => setSettings(s => ({ ...s, topP: v }))}
                max={1}
                step={0.1}
              >
                <Slider.Track style={styles.sliderTrack}>
                  <Slider.Range style={styles.sliderRange} />
                </Slider.Track>
                <Slider.Thumb style={styles.sliderThumb} />
              </Slider.Root>
            </div>

            <div style={styles.dialogButtons}>
              <button 
                style={styles.dialogButton}
                onClick={async () => {
                  await window.api.saveSettings(settings);
                  setSettingsOpen(false);
                }}
              >
                Save
              </button>
              <button 
                style={styles.dialogButtonCancel}
                onClick={() => setSettingsOpen(false)}
              >
                Cancel
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    background: '#1a1a2e',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 20px',
    background: '#16213e',
    borderBottom: '1px solid #0f3460',
  },
  title: {
    fontSize: '18px',
    fontWeight: 600,
    color: '#e94560',
  },
  headerButtons: {
    display: 'flex',
    gap: '10px',
  },
  modeButton: {
    padding: '8px 16px',
    background: '#0f3460',
    border: 'none',
    borderRadius: '6px',
    color: '#eee',
    cursor: 'pointer',
    fontSize: '14px',
  },
  downloadButton: {
    padding: "8px 16px",
    background: "#4CAF50",
    border: "none",
    borderRadius: "6px",
    color: "#fff",
    cursor: "pointer",
    fontSize: "14px",
  },
  settingsButton: {
    padding: '8px 16px',
    background: '#e94560',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '14px',
  },
  main: {
    flex: 1,
    overflow: 'hidden',
  },
  messages: {
    height: '100%',
    overflowY: 'auto',
    padding: '20px',
  },
  emptyState: {
    textAlign: 'center',
    marginTop: '100px',
    color: '#888',
  },
  emptySubtext: {
    fontSize: '14px',
    marginTop: '8px',
  },
  userMessage: {
    display: 'flex',
    justifyContent: 'flex-end',
    marginBottom: '16px',
  },
  assistantMessage: {
    display: 'flex',
    flexDirection: 'column',
    marginBottom: '16px',
  },
  messageContent: {
    maxWidth: '70%',
    padding: '12px 16px',
    borderRadius: '12px',
    lineHeight: 1.5,
  },
  feedbackButtons: {
    display: 'flex',
    gap: '8px',
    marginTop: '4px',
  },
  feedbackButton: {
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    fontSize: '16px',
    padding: '4px',
    opacity: 0.7,
  },
  loadingDots: {
    color: '#888',
  },
  footer: {
    display: 'flex',
    gap: '10px',
    padding: '16px 20px',
    background: '#16213e',
    borderTop: '1px solid #0f3460',
  },
  input: {
    flex: 1,
    padding: '12px 16px',
    background: '#0f3460',
    border: 'none',
    borderRadius: '8px',
    color: '#eee',
    fontSize: '14px',
    outline: 'none',
  },
  sendButton: {
    padding: '12px 24px',
    background: '#e94560',
    border: 'none',
    borderRadius: '8px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '14px',
  },
  dialogOverlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0, 0, 0, 0.7)',
  },
  dialogContent: {
    position: 'fixed',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    background: '#16213e',
    padding: '24px',
    borderRadius: '12px',
    width: '400px',
    maxWidth: '90vw',
  },
  dialogTitle: {
    fontSize: '20px',
    fontWeight: 600,
    marginBottom: '20px',
    color: '#eee',
  },
  settingGroup: {
    marginBottom: '20px',
  },
  settingLabel: {
    display: 'block',
    marginBottom: '8px',
    fontSize: '14px',
    color: '#ccc',
  },
  slider: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    width: '100%',
    height: '20px',
  },
  sliderTrack: {
    position: 'relative',
    flexGrow: 1,
    height: '4px',
    background: '#0f3460',
    borderRadius: '2px',
  },
  sliderRange: {
    position: 'absolute',
    height: '100%',
    background: '#e94560',
    borderRadius: '2px',
  },
  sliderThumb: {
    display: 'block',
    width: '16px',
    height: '16px',
    background: '#fff',
    borderRadius: '50%',
    cursor: 'pointer',
  },
  dialogButtons: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '10px',
    marginTop: '20px',
  },
  dialogButton: {
    padding: '10px 20px',
    background: '#e94560',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    cursor: 'pointer',
  },
  dialogButtonCancel: {
    padding: '10px 20px',
    background: '#0f3460',
    border: 'none',
    borderRadius: '6px',
    color: '#eee',
    cursor: 'pointer',
  },
  dropdownContent: {
    background: '#16213e',
    borderRadius: '8px',
    padding: '8px',
    minWidth: '200px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
  },
  dropdownItem: {
    padding: '10px 12px',
    cursor: 'pointer',
    borderRadius: '4px',
  },
  modeItemName: {
    fontSize: '14px',
    fontWeight: 500,
    color: '#eee',
  },
  modeItemDesc: {
    fontSize: '12px',
    color: '#888',
    marginTop: '2px',
  },
};

export default App;