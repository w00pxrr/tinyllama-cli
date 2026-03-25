"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const child_process_1 = require("child_process");
const path_1 = __importDefault(require("path"));
const electron_log_1 = __importDefault(require("electron-log"));
// @ts-ignore
electron_log_1.default.info('Application starting...');
let mainWindow = null;
let pythonProcess = null;
const isDev = process.env.NODE_ENV !== 'production';
function startPythonBackend() {
    const backendPath = isDev
        ? path_1.default.join(__dirname, '..', '..', 'tinyllama_gui_backend.py')
        : path_1.default.join(__dirname, 'tinyllama_gui_backend.py');
    electron_log_1.default.info('Starting Python backend from:', backendPath);
    pythonProcess = (0, child_process_1.spawn)('python3', [backendPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env }
    });
    pythonProcess.stdout?.on('data', (data) => {
        electron_log_1.default.info('Python stdout:', data.toString());
    });
    pythonProcess.stderr?.on('data', (data) => {
        electron_log_1.default.error('Python stderr:', data.toString());
    });
    pythonProcess.on('error', (err) => {
        electron_log_1.default.error('Failed to start Python backend:', err);
    });
    pythonProcess.on('exit', (code) => {
        electron_log_1.default.info('Python backend exited with code:', code);
    });
}
function createWindow() {
    mainWindow = new electron_1.BrowserWindow({
        width: 900,
        height: 700,
        webPreferences: {
            preload: path_1.default.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
        backgroundColor: '#1a1a2e',
    });
    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    }
    else {
        mainWindow.loadFile(path_1.default.join(__dirname, '..', 'renderer', 'index.html'));
    }
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}
// IPC Handlers
electron_1.ipcMain.handle('send-message', async (_event, message) => {
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
            }
            catch {
                // Continue accumulating
            }
        };
        pythonProcess.stdout?.on('data', onData);
        pythonProcess.stdin.write(JSON.stringify({ type: 'chat', message }) + '\n');
    });
});
electron_1.ipcMain.handle('get-modes', async () => {
    return {
        modes: [
            { id: 'chat', name: 'Chat', description: 'General conversation mode' },
            { id: 'code', name: 'Code', description: 'Programming assistance mode' },
            { id: 'reasoning', name: 'Reasoning', description: 'Logical reasoning mode' }
        ]
    };
});
electron_1.ipcMain.handle('set-mode', async (_event, modeId) => {
    if (!pythonProcess || !pythonProcess.stdin) {
        return { error: 'Backend not running' };
    }
    pythonProcess.stdin.write(JSON.stringify({ type: 'set-mode', mode: modeId }) + '\n');
    return { success: true };
});
electron_1.ipcMain.handle('get-settings', async () => {
    return {
        settings: {
            temperature: 0.7,
            maxTokens: 512,
            topP: 0.9
        }
    };
});
electron_1.ipcMain.handle('save-settings', async (_event, settings) => {
    electron_log_1.default.info('Saving settings:', settings);
    return { success: true };
});
electron_1.ipcMain.handle('feedback', async (_event, type, message) => {
    electron_log_1.default.info(`User feedback: ${type} for message: ${message}`);
    return { success: true };
});
electron_1.ipcMain.handle('select-model', async () => {
    const result = await electron_1.dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
        title: 'Select Model Directory'
    });
    return result.canceled ? null : result.filePaths[0];
});
electron_1.ipcMain.handle('download-model', async () => {
    if (!pythonProcess || !pythonProcess.stdin) {
        return { error: 'Backend not running' };
    }
    pythonProcess.stdin.write(JSON.stringify({ type: 'download-model' }) + '\n');
    return { success: true, message: 'Model download started' };
});
electron_1.app.whenReady().then(() => {
    startPythonBackend();
    createWindow();
});
electron_1.app.on('window-all-closed', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
    if (process.platform !== 'darwin') {
        electron_1.app.quit();
    }
});
electron_1.app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});
