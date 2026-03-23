#!/usr/bin/env node
/**
 * TinyLlama CLI - Terminal chat interface
 */

import readline from 'readline';
import { getClient } from './client.js';
import { AVAILABLE_MODELS, DEFAULT_CONFIG, type ChatMessage } from './types.js';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: '\x1b[36mYou:\x1b[0m ',
});

// Colors
const C = {
  reset: '\x1b[0m',
  cyan: '\x1b[36m',
  cyanBold: '\x1b[1;36m',
  green: '\x1b[32m',
  greenBold: '\x1b[1;32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
};

function log(msg: string, color: keyof typeof C = 'reset') {
  console.log(`${C[color]}${msg}${C.reset}`);
}

function section(title: string) {
  console.log(`\n${C.cyanBold}== ${title} ==${C.reset}`);
}

class CLI {
  private messages: ChatMessage[] = [];
  private modelId: string;
  private loaded = false;

  constructor(modelId: string = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0') {
    this.modelId = modelId;
  }

  async start() {
    section('TinyLlama CLI');
    log('Initializing...', 'cyan');

    try {
      const client = getClient(this.modelId);
      log(`Loading model: ${this.modelId}`, 'yellow');
      await client.load();
      this.loaded = true;
      log('Ready!', 'green');
    } catch (err) {
      log(`Failed to load: ${(err as Error).message}`, 'red');
      process.exit(1);
    }

    this.showHelp();
    this.run();
  }

  showHelp() {
    section('Commands');
    console.log(`
${C.cyan}/help${C.reset}   - Show help
${C.cyan}/clear${C.reset}  - Clear chat
${C.cyan}/save${C.reset}   - Save transcript
${C.cyan}/exit${C.reset}  - Quit
${C.cyan}/models${C.reset} - List available models

${C.yellow}Tips:${C.reset}
  - Model auto-tunes based on your input
  - Ctrl+C to quit
`);
  }

  async generate(input: string): Promise<string> {
    const client = getClient(this.modelId);
    const userMsg: ChatMessage = { role: 'user', content: input };
    this.messages.push(userMsg);

    log('Thinking...', 'cyan');

    try {
      const response = await client.generate(
        this.messages,
        'You are a helpful AI assistant.',
        DEFAULT_CONFIG
      );
      const asstMsg: ChatMessage = { role: 'assistant', content: response };
      this.messages.push(asstMsg);
      return response;
    } catch (err) {
      return `Error: ${(err as Error).message}`;
    }
  }

  handleCommand(cmd: string): boolean {
    const c = cmd.trim().toLowerCase();

    switch (c) {
      case '/help':
        this.showHelp();
        break;
      case '/clear':
        this.messages = [];
        log('Cleared!', 'green');
        break;
      case '/save':
        const fs = require('fs');
        const dir = './transcripts';
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        fs.writeFileSync(`${dir}/chat-${stamp}.json`, JSON.stringify(this.messages, null, 2));
        log(`Saved to transcripts/chat-${stamp}.json`, 'green');
        break;
      case '/exit':
      case '/quit':
        log('Goodbye!', 'cyan');
        process.exit(0);
        break;
      case '/models':
        section('Available Models');
        AVAILABLE_MODELS.forEach(m => {
          console.log(`  ${C.cyanBold}${m.name}${C.reset} - ${m.description}`);
        });
        break;
      default:
        return false;
    }
    return true;
  }

  run() {
    rl.prompt();

    rl.on('line', async (input) => {
      const trimmed = input.trim();
      if (!trimmed) {
        rl.prompt();
        return;
      }

      if (trimmed.startsWith('/')) {
        if (!this.handleCommand(trimmed)) {
          log(`Unknown: ${trimmed}`, 'red');
        }
        rl.prompt();
        return;
      }

      const response = await this.generate(trimmed);
      console.log(`\n${C.greenBold}Assistant:${C.reset}`);
      console.log(response);
      console.log();
      rl.prompt();
    });

    rl.on('close', () => {
      log('\nGoodbye!', 'cyan');
      process.exit(0);
    });
  }
}

// Run CLI
const model = process.argv[2] || 'TinyLlama/TinyLlama-1.1B-Chat-v1.0';
new CLI(model).start().catch(console.error);