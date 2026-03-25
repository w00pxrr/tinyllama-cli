// Type declarations for Electron global variables in main process

declare namespace Electron {
  interface App {
    whenReady(): Promise<void>;
    on(event: string, listener: Function): void;
    quit(): void;
  }

  interface WebContents {
    openDevTools(): void;
  }

  interface BrowserWindow {
    loadURL(url: string): void;
    loadFile(filePath: string): void;
    webContents: WebContents;
    on(event: string, listener: Function): void;
  }

  interface IpcMain {
    handle(channel: string, handler: Function): void;
  }

  interface Dialog {
    showOpenDialog(window: BrowserWindow, options: any): Promise<{
      canceled: boolean;
      filePaths: string[];
    }>;
  }
}

declare const app: Electron.App;
declare const BrowserWindow: {
  new (options?: any): Electron.BrowserWindow;
};
declare const ipcMain: Electron.IpcMain;
declare const dialog: Electron.Dialog;