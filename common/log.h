#pragma once

#include <map>
#include <memory>

#include <spdlog/common.h>
#include <spdlog/spdlog.h>


std::shared_ptr<spdlog::logger> g_logger; // global logger instance
static const spdlog::level::level_enum default_log_level = spdlog::level::info;
extern spdlog::level::level_enum g_log_level;

#define LOG_CLR_TO_EOL "\033[K\r"
#define LOG_COL_DEFAULT "\033[0m"
#define LOG_COL_BOLD "\033[1m"
#define LOG_COL_RED "\033[31m"
#define LOG_COL_GREEN "\033[32m"
#define LOG_COL_YELLOW "\033[33m"
#define LOG_COL_BLUE "\033[34m"
#define LOG_COL_MAGENTA "\033[35m"
#define LOG_COL_CYAN "\033[36m"
#define LOG_COL_WHITE "\033[37m"

namespace lhm {

enum log_colors {
  LOG_COLORS_AUTO = -1,
  LOG_COLORS_DISABLED = 0,
  LOG_COLORS_ENABLED = 1,
};

void init_logger();

static std::map<std::string, spdlog::level::level_enum> log_level_map = {
    {"trace", spdlog::level::trace},
    {"debug", spdlog::level::debug},
    {"info", spdlog::level::info},
    {"warn", spdlog::level::warn},
    {"error", spdlog::level::err},
    {"critical", spdlog::level::critical},
    {"off", spdlog::level::off}
};

static spdlog::level::level_enum get_log_level(const std::string & level_str) {
    for (const auto & pair : log_level_map) {
        if (pair.first == level_str) {
            return pair.second;
        }
    }
    return default_log_level;
}

}

template<spdlog::level::level_enum Lvl, typename... Args>
void lhm_log_impl(fmt::format_string<Args...> fmt, Args&&... args)
{
    if constexpr (Lvl < spdlog::level::n_levels) {
        if (Lvl >= g_log_level && g_logger) {
            if constexpr (Lvl == spdlog::level::trace) {
                g_logger->trace(fmt, std::forward<Args>(args)...);
            } else if constexpr (Lvl == spdlog::level::debug) {
                g_logger->debug(fmt, std::forward<Args>(args)...);
            } else if constexpr (Lvl == spdlog::level::info) {
                g_logger->info(fmt, std::forward<Args>(args)...);
            } else if constexpr (Lvl == spdlog::level::warn) {
                g_logger->warn(fmt, std::forward<Args>(args)...);
            } else if constexpr (Lvl == spdlog::level::err) {
                g_logger->error(fmt, std::forward<Args>(args)...);
            } else if constexpr (Lvl == spdlog::level::critical) {
                g_logger->critical(fmt, std::forward<Args>(args)...);
            }
        }
    }
}

#define LOG_TRACE(...) lhm_log_impl<spdlog::level::trace>(__VA_ARGS__)
#define LOG_DEBUG(...) lhm_log_impl<spdlog::level::debug>(__VA_ARGS__)
#define LOG_INFO(...) lhm_log_impl<spdlog::level::info>(__VA_ARGS__)
#define LOG_WARN(...) lhm_log_impl<spdlog::level::warn>(__VA_ARGS__)
#define LOG_ERROR(...) lhm_log_impl<spdlog::level::err>(__VA_ARGS__)
#define LOG_CRIT(...) lhm_log_impl<spdlog::level::critical>(__VA_ARGS__)
