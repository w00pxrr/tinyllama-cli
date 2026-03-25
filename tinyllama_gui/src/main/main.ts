// Import only non-Electron modules
import { spawn } from 'child_process';
import path from 'path';
import log from 'electron-log';

// In Electron main process, these are available as built-in globals
// @ts-ignore - These are provided by Electron runtime
declare const app: Electron.App;
declare const BrowserWindow: typeof Electron.BrowserWindow;
declare const ipcMain: Electron.IpcMain;
declare const dialog: Electron.Dialog;

// Debug
console.log('process.versions.electron:', process.versions.electron);
console.log('app available:', !!app);
console.log('BrowserWindow available:', !!BrowserWindow);
console.log('ipcMain available:', !!ipcMain);
console.log('dialog available:', !!dialog);

log.info('Application starting...');

let mainWindow: Electron.BrowserWindow | null = null;
let pythonProcess: any = null;

const isDev = process.env.NODE_ENV !== 'production';

function startPythonBackend(): void {
  const backendPath = isDev 
    ? path.join(__dirname, '..', '..', 'tinyllama_gui_backend.py')
    : path.join(__dirname, 'tinyllama_gui_backend.py');

  log.info('Starting Python backend from:', backendPath);
  pythonProcess = spawn('python3', [backendPath], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env }
  });

  pythonProcess.stdout?.on('data', (data: Buffer) => {
    log.info('Python stdout:', data.toString());
  });

  pythonProcess.stderr?.on('data', (data: Buffer) => {
    log.error('Python stderr:', data.toString());
  });

  pythonProcess.on('error', (err: Error) => {
    log.error('Failed to start Python backend:', err);
  });

  pythonProcess.on('exit', (code: number) => {
    log.info('Python backend exited with code:', code);
  });
}

function createWindow(): void {
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
    if (mainWindow) {
      mainWindow.loadURL('http://localhost:5173');
      mainWindow.webContents.openDevTools();
    }
  } else {
    if (mainWindow) {
      mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
    }
  }

  if (mainWindow) {
    mainWindow.on('closed', () => {
      mainWindow = null;
    });
  }
}

// IPC Handlers
ipcMain.handle('send-message', async (_event, message: string) => {
  return new Promise((resolve) => {
    if (!pythonProcess || !pythonProcess.stdin) {
      resolve({ error: 'Backend not running' });
      return;
    }

    let response = '';
    const timeout = setTimeout(() => {
      resolve({ error: 'Timeout waiting for response' });
    }, 60000);

    const onData = (data: Buffer) => {
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

ipcMain.handle('set-mode', async (_event, modeId: string) => {
  // In a real app, this would update the backend configuration
  log.info('Setting mode to:', modeId);
  return { success: true };
});

ipcMain.handle('get-settings', async () => {
  // Return default settings
  return {
    temperature: 0.7,
    maxTokens: 2048,
    topP: 0.9,
    systemPrompt: "You are a helpful AI assistant."
  };
});

ipcMain.handle('save-settings', async (_event, settings: any) => {
  // In a real app, this would save settings to disk
  log.info('Saving settings:', settings);
  return { success: true };
});

ipcMain.handle('feedback', async (_event, type: 'like' | 'dislike', message: string) => {
  log.info(`Received ${type} feedback for message:`, message);
  // In a real app, this would be used for training/improvement
  return { success: true };
});

ipcMain.handle('select-model', async () => {
  if (!mainWindow) {
    return null;
  }
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
  // Open a dialog to let user select download option
  if (!mainWindow) {
    return { success: false, message: 'No window available' };
  }
  const result = await dialog.showMessageBox(mainWindow, {
    type: 'info',
    title: 'Download Model',
    message: 'Download a TinyLlama model?',
    detail: 'This will open the model download script. You can choose TinyLlama, SmolLM2, or other models.',
    buttons: ['Download TinyLlama', 'Cancel'],
    defaultId: 0
  });
  
  if (result.response === 0) {
    // Spawn download script
    const downloadPath = path.join(__dirname, '..', '..', 'download_model.py');
    spawn('python3', [downloadPath], {
      stdio: 'inherit',
      shell: true
    });
    return { success: true, message: 'Opening download script...' };
  }
  
  return { success: false, message: 'Download cancelled' };
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
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});