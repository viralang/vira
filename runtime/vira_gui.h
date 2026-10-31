#ifndef VIRA_GUI_H
#define VIRA_GUI_H

#include "vira_rt.h"
#include "vira_gui_types.h"

// ==============================================================================
// NATIVE OS & HARDWARE ACCELERATED BACKENDS
// ==============================================================================

#if defined(VIRA_USE_RAYLIB)
    #include "vira_gui_raylib.h"
#elif defined(_WIN32)
    #include "vira_gui_win32.h"
#elif (defined(__linux__) || defined(__unix__)) && !defined(VIRA_NO_X11)
    #include "vira_gui_x11.h"
#elif defined(__APPLE__) && defined(__OBJC__)
    #include "vira_gui_cocoa.h"
#else
    #include "vira_gui_headless.h"
#endif

// ==============================================================================
// SHARED WIDGETS & DECLARATIVE AUTO-LAYOUT ENGINE
// ==============================================================================
#include "vira_gui_widgets.h"

#endif // VIRA_GUI_H