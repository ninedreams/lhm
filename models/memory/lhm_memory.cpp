#include "lhm_memory.h"

lhm_memory_status lhm_memory_status_combine(lhm_memory_status s0, lhm_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LHM_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LHM_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LHM_MEMORY_STATUS_FAILED_PREPARE:
        case LHM_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LHM_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LHM_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LHM_MEMORY_STATUS_FAILED_PREPARE:
        case LHM_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LHM_MEMORY_STATUS_SUCCESS : LHM_MEMORY_STATUS_NO_UPDATE;
}

bool lhm_memory_status_is_fail(lhm_memory_status status) {
    switch (status) {
        case LHM_MEMORY_STATUS_SUCCESS:
        case LHM_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LHM_MEMORY_STATUS_FAILED_PREPARE:
        case LHM_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
