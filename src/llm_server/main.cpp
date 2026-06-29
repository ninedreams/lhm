#include "config.h"
#include "common.h"
#include "common_params.h"
#include "log.h"
#include "server.h"

using namespace lhm;

int main(int argc, char ** argv) {
    // Save original argc/argv before gflags modifies them
    int orig_argc = argc;
    char ** orig_argv = argv;

    init_config(argc, argv);

    common_params params;
    fill_common_params(params);

    init_logger("server");

    return llm_server(params, orig_argc, orig_argv);
}
