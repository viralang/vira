const path = require('path');
const fs = require('fs');
const os = require('os');
const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');

let client;

function resolveVexBinary() {
    const config = vscode.workspace.getConfiguration('vex');
    const customPath = config.get('compilerPath');
    if (customPath && customPath !== 'vex' && fs.existsSync(customPath)) {
        return customPath;
    }

    // Check user home default install directory (~/.vex/bin/vex)
    const home = os.homedir();
    const isWin = os.platform() === 'win32';
    const homeBinary = path.join(home, '.vex', 'bin', isWin ? 'vex.exe' : 'vex');
    if (fs.existsSync(homeBinary)) {
        return homeBinary;
    }

    // Default to 'vex' in PATH
    return isWin ? 'vex.exe' : 'vex';
}

function activate(context) {
    const vexBin = resolveVexBinary();

    // Configure the Language Server process (`vex lsp` via stdio)
    const serverOptions = {
        command: vexBin,
        args: ['lsp'],
        transport: TransportKind.stdio
    };

    // Configure language client for Vex source files
    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'vex' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.vex')
        }
    };

    // Create the language client and start the server
    client = new LanguageClient(
        'vexls',
        'Vex Language Server',
        serverOptions,
        clientOptions
    );

    // Register restart command
    const restartCmd = vscode.commands.registerCommand('vex.restartServer', async () => {
        if (client) {
            await client.stop();
        }
        client.start();
        vscode.window.showInformationMessage('Vex Language Server restarted.');
    });
    context.subscriptions.push(restartCmd);

    // Start the client (spawns `vex lsp`)
    client.start().catch((err) => {
        vscode.window.showWarningMessage(
            `Could not start Vex Language Server (${vexBin}): ${err.message}. Please install Vex or configure 'vex.compilerPath'.`
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
