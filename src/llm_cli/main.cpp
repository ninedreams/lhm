#include "config.h"
#include "log.h"
#include "cli.h"

using namespace lhm;

int main(int argc, char ** argv) {
    init_config(argc, argv);
    
    init_logger("cli");

    return llm_cli();
}
