const path = require('path');
const fs = require('fs');
const os = require('os');
const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');

let client;

function resolveViraBinary() {
    const config = vscode.workspace.getConfiguration('vira');
    const customPath = config.get('compilerPath');
    if (customPath && customPath !== 'vira' && fs.existsSync(customPath)) {
        return customPath;
    }

    // Check user home default install directory (~/.vira/bin/vira or ~/.vex/bin/vex)
    const home = os.homedir();
    const isWin = os.platform() === 'win32';
    const viraBinary = path.join(home, '.vira', 'bin', isWin ? 'vira.exe' : 'vira');
    if (fs.existsSync(viraBinary)) {
        return viraBinary;
    }
    const vexBinary = path.join(home, '.vex', 'bin', isWin ? 'vex.exe' : 'vex');
    if (fs.existsSync(vexBinary)) {
        return vexBinary;
    }

    // Default to 'vira' in PATH
    return isWin ? 'vira.exe' : 'vira';
}

function activate(context) {
    const viraBin = resolveViraBinary();

    // Configure the Language Server process (`vira lsp` via stdio)
    const serverOptions = {
        command: viraBin,
        args: ['lsp'],
        transport: TransportKind.stdio
    };

    // Configure language client for Vira source files
    const clientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'vira' },
            { scheme: 'file', language: 'vex' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{vir,vex}')
        }
    };

    // Create the language client and start the server
    client = new LanguageClient(
        'virals',
        'Vira Language Server',
        serverOptions,
        clientOptions
    );

    // Register restart command
    const restartCmd = vscode.commands.registerCommand('vira.restartServer', async () => {
        if (client) {
            await client.stop();
        }
        client.start();
        vscode.window.showInformationMessage('Vira Language Server restarted.');
    });
    context.subscriptions.push(restartCmd);

    // Start the client (spawns `vira lsp`)
    client.start().catch((err) => {
        vscode.window.showWarningMessage(
            `Could not start Vira Language Server (${vexBin}): ${err.message}. Please install Vira or configure 'vira.compilerPath'.`
        );
    });
}

function deactivate() {
    if (!client) {
        return undefined;
    }
    return client.stop();
}

module.exports = {
    activate,
    deactivate
};
