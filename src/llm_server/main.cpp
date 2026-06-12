#include "common/config.h"
#include "log.h"
#include "server.h"

using namespace lhm;

int main(int argc, char ** argv) {
    init_config(argc, argv);
    
    init_logger();

    return llm_server();
}
