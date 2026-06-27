#pragma once

#include <cstdarg>
#include <cstdlib>
#include <cassert>

#include <fmt/core.h>
#include <fmt/format.h>

#include "log.h"

namespace lhm {

template<typename... Args>
[[noreturn]] static void abort(const char* file, int line, 
                               fmt::format_string<Args...> fmt, Args&&... args) {
    std::string content = fmt::format(fmt, std::forward<Args>(args)...);
    LOG_CRIT("{}:{}: {}", file, line, content);
    std::abort();
}

}

#define LHM_ABORT(...)                                                  \
    do {                                                                \
        lhm::abort(__FILE__, __LINE__, __VA_ARGS__);                    \
    } while (0)

#define LHM_ASSERT(x)                                                   \
    do {                                                                \
        if (!(x)) {                                                     \
            LHM_ABORT("LHM_ASSERT(%s) failed", #x);                     \
        }                                                               \
    } while (0)

