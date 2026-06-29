#pragma once

#include "common.h"
#include "config.h"
#include "server_common.h"
#include "server_http.h"
#include "server_queue.h"

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <set>

/**
 * state diagram:
 *
 * UNLOADED ──► LOADING ──► LOADED ◄──── SLEEPING
 *  ▲            │            │               ▲
 *  └───failed───┘            │               │
 *  ▲                         └──sleeping─────┘
 *  └────────unloaded─────────┘
 */
enum server_model_status {
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED,
    SERVER_MODEL_STATUS_SLEEPING
};

enum server_model_source {
    SERVER_MODEL_SOURCE_PRESET,
    SERVER_MODEL_SOURCE_MODELS_DIR,
};

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_UNLOADED:    return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:     return "loading";
        case SERVER_MODEL_STATUS_LOADED:      return "loaded";
        case SERVER_MODEL_STATUS_SLEEPING:    return "sleeping";
        default:                              return "unknown";
    }
}

// Simple model preset: a name + key-value options map
// Replaces the old common_preset which depended on arg.h/common_arg
struct model_preset {
    std::string name;
    std::map<std::string, std::string> options; // key (env-style) -> value

    void set_option(const std::string & key, const std::string & value) {
        options[key] = value;
    }

    void unset_option(const std::string & key) {
        options.erase(key);
    }

    bool get_option(const std::string & key, std::string & value) const {
        auto it = options.find(key);
        if (it != options.end()) {
            value = it->second;
            return true;
        }
        return false;
    }

    void merge(const model_preset & other) {
        for (const auto & [k, v] : other.options) {
            options[k] = v;
        }
    }

    // Convert preset options to CLI argument list
    std::vector<std::string> to_args(const std::string & bin_path = "") const;

    // Apply a subset of options (identified by keys) to common_params for model path resolution
    void apply_model_options(common_params & params, const std::set<std::string> & keys) const;
};

using model_presets = std::map<std::string, model_preset>;

struct server_model_meta {
    server_model_source source = SERVER_MODEL_SOURCE_PRESET;
    model_preset preset;
    std::string name;
    std::set<std::string> aliases; // additional names that resolve to this model
    std::set<std::string> tags;    // informational tags, not used for routing
    int port = 0;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0; // for LRU unloading
    std::vector<std::string> args; // args passed to the model instance, will be populated by render_args()
    json loaded_info; // info to be reflected via /v1/models endpoint
    int exit_code = 0; // exit code of the model instance process (only valid if status == FAILED)
    int stop_timeout = 0; // seconds to wait before force-killing the model instance during shutdown

    bool is_ready() const {
        return status == SERVER_MODEL_STATUS_LOADED;
    }

    bool is_running() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING || status == SERVER_MODEL_STATUS_SLEEPING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }

    void update_args(std::string bin_path);
    void update_caps();
};
struct server_subproc; // defined in server_models.cpp

struct server_models {

private:
    struct instance_t {
        std::shared_ptr<server_subproc> subproc; // shared between main thread and monitoring thread
        std::thread th;
        server_model_meta meta;
        FILE * stdin_file = nullptr;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    // for stopping models
    std::condition_variable cv_stop;
    std::set<std::string> stopping_models;

    // set to true while load_models() is executing a reload; load() will wait until clear
    bool is_reloading = false;

    // if true, the next get_meta() will trigger a reload of model list
    bool need_reload = false;

    common_params base_params;
    std::string bin_path;
    std::vector<std::string> base_env;
    model_preset base_preset; // base preset from lhm_server CLI args

    void update_meta(const std::string & name, const server_model_meta & meta);

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

    // notify SSE clients
    void notify_sse(const std::string & event, const std::string & model_id, const json & data = nullptr);

public:
    server_models(const common_params & params, int argc, char ** argv);

    server_response sse; // for real-time updates via SSE endpoint

    // (re-)load the list of models from various sources and prepare the metadata mapping
    // - if this is called the first time, simply populate the metadata
    // - if this is called subsequently (e.g. when refreshing from disk):
    //   - if a model is running but updated or removed from the source, it will be unloaded
    //   - if a model is not running, it will be added or updated according to the source
    void load_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta> get_all_meta();

    // load and unload model instances
    // these functions are thread-safe
    void load(const std::string & name);
    void unload(const std::string & name);
    void unload_all();

    // update the status of a model instance (thread-safe)
    void update_status(const std::string & name, server_model_status status, int exit_code);
    void update_loaded_info(const std::string & name, std::string & raw_info);

    // remove a model from the list (thread-safe)
    // returns false if the model doesn't exist
    bool remove(const std::string & name);

    // wait until the model instance is fully loaded (thread-safe)
    // note: predicate is called while holding the lock
    // return when the model no longer in "loading" state
    void wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate);
    void wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate);

    // ensure the model is in ready state (thread-safe)
    // return false if model is ready
    // otherwise, load the model and blocking wait until it's ready, then return true (meta may need to be refreshed)
    bool ensure_model_ready(const std::string & name);

    // return true if the current process is a child server instance
    static bool is_child_server();

    // notify the router server that a model instance is ready
    // return the monitoring thread (to be joined by the caller)
    static std::thread setup_child_server(const std::function<void(int)> & shutdown_handler, const json & model_info);

    // notify the router server that the sleeping state has changed
    static void notify_router_sleeping_state(bool sleeping);
};
