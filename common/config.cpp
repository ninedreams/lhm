#include "common/config.h"

namespace lhm {

void init_config(int argc, char ** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}

} // namespace lhm