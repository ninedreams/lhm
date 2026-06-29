#pragma once

// Forwarding header: redirects nlohmann/json.hpp to the RapidJSON compatibility layer.
// This file is found BEFORE the original json/nlohmann/json.hpp thanks to
// the include_directories order in CMakeLists.txt.

#include "rj_json.h"
