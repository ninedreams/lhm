#pragma once

#include <cstdarg>
#include <cassert>

#include <fmt/core.h>

#include "log.h"

namespace lhm {

static void abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::string content = fmt::vformat(fmt, fmt::make_format_args(args));
    va_end(args);

    LOG_CRIT("{}:{}: {}", file, line, content);

    std::abort();
}

}

#define LHM_ABORT(...)                                                  \
    do {                                                                \
        lhm::abort(__FILE__, __LINE__, __VA_ARGS__);                \
    } while (0)

#define LHM_ASSERT(x)                                                   \
    do {                                                                \
        if (!(x)) {                                                     \
            LHM_ABORT("LHM_ASSERT(%s) failed", #x);                     \
        }                                                               \
    } while (0)
