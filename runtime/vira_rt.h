#ifndef VIRA_RT_H
#define VIRA_RT_H

// Modular Vira C99 Runtime Umbrella Header
// Subdirectories under runtime/rt/ isolate each runtime subsystem while
// preserving single-include (`#include "vira_rt.h"`) compilation semantics.

#include "rt/platform/platform.h"
#include "rt/string/string.h"
#include "rt/slice/slice.h"
#include "rt/arena/arena.h"

#include "rt/tensor/core.h"
#include "rt/tensor/quant_registry.h"
#include "rt/tensor/quant_simd.h"
#include "rt/tensor/ops.h"
#include "rt/tensor/autograd.h"

#include "rt/scheduler/scheduler.h"
#include "rt/io/io.h"
#include "rt/net/net.h"
#include "rt/json/json.h"

#include "rt/weights/bin.h"
#include "rt/weights/safetensors.h"
#include "rt/weights/gguf.h"

#include "rt/llm/llm.h"
#include "rt/stdlib/stdlib.h"
#include "rt/sys/sys.h"

#include "rt/platform/telemetry.h"

#include "vira_gpu.h"
#include "vira_gui.h"

#endif // VIRA_RT_H
