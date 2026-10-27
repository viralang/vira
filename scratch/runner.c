#include <windows.h>
#include <stdio.h>

int main(void) {
    for (int i = 1; i <= 20; i++) {
        STARTUPINFOA si = { sizeof(si) };
        PROCESS_INFORMATION pi;
        if (!CreateProcessA(NULL, "check_disp_opt.exe", NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
            printf("CreateProcess failed\n");
            return 1;
        }
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD exitCode = 0;
        GetExitCodeProcess(pi.hProcess, &exitCode);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        if (exitCode != 0) {
            printf("FAILED on run %d with exit code 0x%08lX (%ld)\n", i, exitCode, (long)exitCode);
            return 1;
        }
        printf("Run %d: SUCCESS\n", i);
    }
    printf("ALL 20 RUNS SUCCEEDED!\n");
    return 0;
}
