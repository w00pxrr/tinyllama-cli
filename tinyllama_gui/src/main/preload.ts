import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('api', {
  sendMessage: (message: string) => ipcRenderer.invoke('send-message', message),
  getModes: () => ipcRenderer.invoke('get-modes'),
  setMode: (modeId: string) => ipcRenderer.invoke('set-mode', modeId),
  getSettings: () => ipcRenderer.invoke('get-settings'),
  saveSettings: (settings: object) => ipcRenderer.invoke('save-settings', settings),
  feedback: (type: 'like' | 'dislike', message: string) => ipcRenderer.invoke('feedback', type, message),
  selectModel: () => ipcRenderer.invoke('select-model'),
  downloadModel: () => ipcRenderer.invoke('download-model'),
});