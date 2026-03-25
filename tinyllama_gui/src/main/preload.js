const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  sendMessage: (message) => ipcRenderer.invoke('send-message', message),
  getModes: () => ipcRenderer.invoke('get-modes'),
  setMode: (modeId) => ipcRenderer.invoke('set-mode', modeId),
  getSettings: () => ipcRenderer.invoke('get-settings'),
  saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),
  feedback: (type, message) => ipcRenderer.invoke('feedback', type, message),
  selectModel: () => ipcRenderer.invoke('select-model'),
  downloadModel: () => ipcRenderer.invoke('download-model'),
});