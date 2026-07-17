#pragma once

#include <cstdarg>

#include <map>
#include <memory>
#include <string>

#include <fmt/core.h>
#include <fmt/format.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

static const spdlog::level::level_enum default_log_level = spdlog::level::info;

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

inline std::shared_ptr<spdlog::logger> g_logger; // global logger instance
inline spdlog::level::level_enum g_log_level;

enum log_colors {
  LOG_COLORS_AUTO = -1,
  LOG_COLORS_DISABLED = 0,
  LOG_COLORS_ENABLED = 1,
};

void init_logger(const std::string& log_name="lhm");

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

void flush_logger();

}

template<spdlog::level::level_enum Lvl, typename... Args>
void lhm_log_impl(const char* file, const char* func, int line,
                  fmt::format_string<Args...> fmt, Args&&... args)
{
    if constexpr (Lvl < spdlog::level::n_levels) {
        if (Lvl >= lhm::g_log_level && lhm::g_logger) {
            auto formatted = fmt::format(fmt, std::forward<Args>(args)...);
            auto msg = fmt::format("[{}:{}:{}] {}", file, func, line, formatted);
            if constexpr (Lvl == spdlog::level::trace) {
                lhm::g_logger->trace(msg);
            } else if constexpr (Lvl == spdlog::level::debug) {
                lhm::g_logger->debug(msg);
            } else if constexpr (Lvl == spdlog::level::info) {
                lhm::g_logger->info(msg);
            } else if constexpr (Lvl == spdlog::level::warn) {
                lhm::g_logger->warn(msg);
            } else if constexpr (Lvl == spdlog::level::err) {
                lhm::g_logger->error(msg);
            } else if constexpr (Lvl == spdlog::level::critical) {
                lhm::g_logger->critical(msg);
            }
        }
    }
}

#define LOG_TRACE(...) lhm_log_impl<spdlog::level::trace>(__FILE__, __func__, __LINE__, __VA_ARGS__)
#define LOG_DEBUG(...) lhm_log_impl<spdlog::level::debug>(__FILE__, __func__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) lhm_log_impl<spdlog::level::info>(__FILE__, __func__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...) lhm_log_impl<spdlog::level::warn>(__FILE__, __func__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) lhm_log_impl<spdlog::level::err>(__FILE__, __func__, __LINE__, __VA_ARGS__)
#define LOG_CRIT(...) lhm_log_impl<spdlog::level::critical>(__FILE__, __func__, __LINE__, __VA_ARGS__)

#define LHM_ABORT(...)                                                  \
    do {                                                                \
        LOG_CRIT(__VA_ARGS__);                                          \
        std::abort();                                                   \
    } while (0)

#define LHM_ASSERT(x)                                                   \
    do {                                                                \
        if (!(x)) {                                                     \
            LHM_ABORT("LHM_ASSERT({}) failed", #x);                     \
        }                                                               \
    } while (0)

#define LHM_ASSERT_MSG(x, msg)                                          \
    do {                                                                \
        if (!(x)) {                                                     \
            LHM_ABORT("LHM_ASSERT({}) failed, msg: {}", #x, msg);       \
        }                                                               \
    } while (0)
