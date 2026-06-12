#pragma once

#include <gflags/gflags.h>

namespace lhm {

void init_config(int argc, char ** argv);

}

// log
DEFINE_string(log_file, "logs/lhm.log", "Log file path");
DEFINE_int32(log_rotate_hour, 3, "Log rotate hour");
DEFINE_int32(log_rotate_minute, 0, "Log rotate minute");
DEFINE_string(log_level, "info", "Log level (trace, debug, info, warn, error, critical, off), if not found set off.");
DEFINE_string(log_pattern, "[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v", "Log pattern (see spdlog documentation for details)");

// llm
DEFINE_bool(enable_reasoning, false,
            "Enable reasoning content in the response, if supported by the "
            "model/template");
DEFINE_string(model_path, "models/Qwen3.5-9B-Q4_K_M.gguf", "model path");

// http
DEFINE_string(host, "0.0.0.0", "Hostname to bind the server to");
DEFINE_int32(port, 8080, "Port to bind the server to");
