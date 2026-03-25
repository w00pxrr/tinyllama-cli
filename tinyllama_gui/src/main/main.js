// Main Electron process
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const log = require('electron-log');

console.log('process.versions.electron:', process.versions.electron);
console.log('process.platform:', process.platform);

log.info('Application starting...');

let mainWindow = null;
let pythonProcess = null;

const isDev = process.env.NODE_ENV !== 'production';

function startPythonBackend() {
  const backendPath = isDev 
    ? path.join(__dirname, '..', '..', 'tinyllama_gui_backend.py')
    : path.join(__dirname, 'tinyllama_gui_backend.py');

  log.info('Starting Python backend from:', backendPath);
  pythonProcess = spawn('python3', [backendPath], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env }
  });

  pythonProcess.stdout?.on('data', (data) => {
    log.info('Python stdout:', data.toString());
  });

  pythonProcess.stderr?.on('data', (data) => {
    log.error('Python stderr:', data.toString());
  });

  pythonProcess.on('error', (err) => {
    log.error('Failed to start Python backend:', err);
  });

  pythonProcess.on('exit', (code) => {
    log.info('Python backend exited with code:', code);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    backgroundColor: '#1a1a2e',
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Handlers
ipcMain.handle('send-message', async (event, message) => {
  return new Promise((resolve) => {
    if (!pythonProcess || !pythonProcess.stdin) {
      resolve({ error: 'Backend not running' });
      return;
    }

    let response = '';
    const timeout = setTimeout(() => {
      resolve({ error: 'Timeout waiting for response' });
    }, 60000);

    const onData = (data) => {
      response += data.toString();
      try {
        const parsed = JSON.parse(response);
        clearTimeout(timeout);
        pythonProcess?.stdout?.removeListener('data', onData);
        resolve(parsed);
      } catch {
        // Continue accumulating
      }
    };

    pythonProcess.stdout?.on('data', onData);
    pythonProcess.stdin.write(JSON.stringify({ type: 'chat', message }) + '\n');
  });
});

ipcMain.handle('get-modes', async () => {
  return {
    modes: [
      { id: 'chat', name: 'Chat', description: 'General conversation mode' },
      { id: 'code', name: 'Code', description: 'Software development assistance' },
      { id: 'creative', name: 'Creative', description: 'Writing and brainstorming' },
      { id: 'analysis', name: 'Analysis', description: 'Logical reasoning and problem solving' }
    ]
  };
});

ipcMain.handle('set-mode', async (event, modeId) => {
  log.info('Setting mode to:', modeId);
  return { success: true };
});

ipcMain.handle('get-settings', async () => {
  return {
    temperature: 0.7,
    maxTokens: 2048,
    topP: 0.9,
    systemPrompt: "You are a helpful AI assistant."
  };
});

ipcMain.handle('save-settings', async (event, settings) => {
  log.info('Saving settings:', settings);
  return { success: true };
});

ipcMain.handle('feedback', async (event, type, message) => {
  log.info(`Received ${type} feedback for message:`, message);
  return { success: true };
});

ipcMain.handle('select-model', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Model Files', extensions: ['bin', 'gguf', 'pt', 'pth'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('download-model', async () => {
  log.info('Opening model download dialog');
  return { success: true, message: 'Model download initiated' };
});

// App lifecycle
app.whenReady().then(() => {
  createWindow();
  startPythonBackend();

  app.on('activate', () => {
    if (mainWindow === null) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});