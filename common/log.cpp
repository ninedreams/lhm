#include <cstdarg>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>

#include "log.h"
#include "config.h"

namespace lhm {

void init_logger() {
    if (!g_logger) {
      g_logger = spdlog::daily_logger_mt(
          "", FLAGS_log_file, FLAGS_log_rotate_hour, FLAGS_log_rotate_minute);
      g_logger->set_pattern(FLAGS_log_pattern);
    }
    g_log_level = get_log_level(FLAGS_log_level);
    g_logger->set_level(g_log_level);
    g_logger->flush_on(spdlog::level::warn);
    spdlog::flush_every(std::chrono::seconds(1));
    g_logger->info("Lhm logger initialized with level: %s", FLAGS_log_level.c_str());
}

}