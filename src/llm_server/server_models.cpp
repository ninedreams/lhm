#include "server_common.h"
#include "server_models.h"

#include "config.h"

#include <httplib.h>
#include <subprocess.h>

#include <functional>
#include <optional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <random>
#include <sstream>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_READY "cmd_child_to_router:ready" // also sent when waking up from sleep
#define CMD_CHILD_TO_ROUTER_SLEEP "cmd_child_to_router:sleep"
#define CMD_CHILD_TO_ROUTER_INFO  "cmd_child_to_router:info:" // followed by json string

// address for child process, this is needed because router may run on 0.0.0.0
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
#define CHILD_ADDR "127.0.0.1"

struct server_subproc {
    std::optional<subprocess_s> sproc;
    std::atomic<bool> stopped{false}; // signal child process exit

    subprocess_s & get() {
        LHM_ASSERT(sproc.has_value() && "subprocess not initialized");
        return sproc.value();
    }

    bool is_alive() {
        return sproc.has_value() && subprocess_alive(&sproc.value());
    }

    void terminate() {
        if (!sproc.has_value()) {
            return;
        }
#if defined(_WIN32)
        if (sproc->hProcess == NULL) {
            return;
        }
#else
        if (sproc->child <= 0) {
            return;
        }
#endif
        subprocess_terminate(&sproc.value());
    }
};


static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}
// ============================================================================
// model_preset method implementations
// ============================================================================

// Map from env-style key to CLI flag name
static const std::map<std::string, std::string> & get_key_to_flag_map() {
    static const std::map<std::string, std::string> m = {
        {"LHM_ARG_MODEL",            "--model"},
        {"LHM_ARG_HOST",             "--host"},
        {"LHM_ARG_PORT",             "--port"},
        {"LHM_ARG_ALIAS",            "--alias"},
        {"LHM_ARG_TAGS",             "--tags"},
        {"LHM_ARG_NGPU",             "--ngl"},
        {"LHM_ARG_CTX_SIZE",         "--ctx_size"},
        {"LHM_ARG_BATCH_SIZE",       "--batch_size"},
        {"LHM_ARG_UBATCH_SIZE",      "--ubatch_size"},
        {"LHM_ARG_THREADS",          "--threads"},
        {"LHM_ARG_TEMP",             "--temp"},
        {"LHM_ARG_TOP_K",            "--top_k"},
        {"LHM_ARG_TOP_P",            "--top_p"},
        {"LHM_ARG_MIN_P",            "--min_p"},
        {"LHM_ARG_REPEAT_PENALTY",   "--repeat_penalty"},
        {"LHM_ARG_SEED",             "--seed"},
        {"LHM_ARG_FLASH_ATTN",       "--flash_attn"},
        {"LHM_ARG_CACHE_TYPE_K",     "--cache_type_k"},
        {"LHM_ARG_CACHE_TYPE_V",     "--cache_type_v"},
        {"LHM_ARG_CONT_BATCHING",    "--cont_batching"},
        {"LHM_ARG_PARALLEL",         "--parallel"},
        {"LHM_ARG_MOE_OFFLOAD",      "--moe_offload"},
        {"LHM_ARG_NO_MMAP",          "--no_mmap"},
        {"LHM_ARG_MLOCK",            "--mlock"},
        {"LHM_ARG_API_KEY",          "--api_key"},
        {"LHM_ARG_SSL_KEY_FILE",     "--ssl_key_file"},
        {"LHM_ARG_SSL_CERT_FILE",    "--ssl_cert_file"},
        {"LHM_ARG_CHAT_TEMPLATE",    "--chat_template"},
        {"LHM_ARG_JINJA",            "--jinja"},
        {"LHM_ARG_NO_KV_OFFLOAD",    "--no_kv_offload"},
        {"LHM_ARG_CACHE_PROMPT",     "--cache_prompt"},
        {"LHM_ARG_REASONING_FORMAT", "--reasoning_format"},
        {"LHM_ARG_REASONING",        "--reasoning"},
        {"LHM_ARG_REASONING_BUDGET", "--reasoning_budget"},
        {"LHM_ARG_SPEC_DRAFT_MODEL", "--spec_draft_model"},
        {"LHM_ARG_SPEC_DRAFT_NGL",   "--spec_draft_ngl"},
        {"LHM_ARG_SPEC_TYPE",        "--spec_type"},
        {"LHM_ARG_LOGIT_BIAS",       "--logit_bias"},
        {"LHM_ARG_GRAMMAR",          "--grammar"},
        {"LHM_ARG_GRAMMAR_FILE",     "--grammar_file"},
        {"LHM_ARG_JSON_SCHEMA",      "--json_schema"},
        {"LHM_ARG_DRY_MULTIPLIER",   "--dry_multiplier"},
        {"LHM_ARG_DRY_BASE",         "--dry_base"},
        {"LHM_ARG_DRY_ALLOWED_LENGTH","--dry_allowed_length"},
        {"LHM_ARG_DRY_PENALTY_LAST_N","--dry_penalty_last_n"},
        {"LHM_ARG_DYNATEMP_RANGE",   "--dynatemp_range"},
        {"LHM_ARG_DYNATEMP_EXP",     "--dynatemp_exp"},
        {"LHM_ARG_MIROSTAT",         "--mirostat"},
        {"LHM_ARG_MIROSTAT_LR",      "--mirostat_lr"},
        {"LHM_ARG_MIROSTAT_ENT",     "--mirostat_ent"},
        {"LHM_ARG_TYPICAL",          "--typical"},
        {"LHM_ARG_PRESENCE_PENALTY", "--presence_penalty"},
        {"LHM_ARG_FREQUENCY_PENALTY","--frequency_penalty"},
        {"LHM_ARG_XTC_PROBABILITY",  "--xtc_probability"},
        {"LHM_ARG_XTC_THRESHOLD",    "--xtc_threshold"},
        {"LHM_ARG_SAMPLERS",         "--samplers"},
        {"LHM_ARG_IGNORE_EOS",       "--ignore_eos"},
        {"LHM_ARG_POOLING",          "--pooling"},
        {"LHM_ARG_ROPE_SCALING",     "--rope_scaling"},
        {"LHM_ARG_ROPE_SCALE",       "--rope_scale"},
        {"LHM_ARG_ROPE_FREQ_BASE",   "--rope_freq_base"},
        {"LHM_ARG_YARN_ORIG_CTX",    "--yarn_orig_ctx"},
        {"LHM_ARG_DEFrag_THOLD",     "--defrag_thold"},
        {"LHM_ARG_RPC",              "--rpc"},
        {"LHM_ARG_DEVICE",           "--device"},
        {"LHM_ARG_OVERRIDE_TENSOR",  "--override_tensor"},
        {"LHM_ARG_NUMA",             "--numa"},
        {"LHM_ARG_TIMEOUT",          "--timeout"},
        {"LHM_ARG_THREADS_HTTP",     "--threads_http"},
        {"LHM_ARG_METRICS",          "--metrics"},
        {"LHM_ARG_SLOTS",            "--slots"},
        {"LHM_ARG_SLOT_SAVE_PATH",   "--slot_save_path"},
        {"LHM_ARG_CACHE_REUSE",      "--cache_reuse"},
        {"LHM_ARG_SIMPLE_IO",        "--simple_io"},
        {"LHM_ARG_SSE_PING_INTERVAL","--sse_ping_interval"},
        {"LHM_ARG_SLOT_PROMPT_SIMILARITY","--slot_prompt_similarity"},
        {"LHM_ARG_NO_CONTEXT_SHIFT", "--no_context_shift"},
        {"LHM_ARG_PREFILL_ASSISTANT","--prefill_assistant"},
        {"LHM_ARG_SKIP_CHAT_PARSING","--skip_chat_parsing"},
        {"LHM_ARG_SLEEP_IDLE_SECONDS","--sleep_idle_seconds"},
        {"LHM_ARG_CHAT_TEMPLATE_FILE","--chat_template_file"},
        {"LHM_ARG_CHAT_TEMPLATE_KWARGS","--chat_template_kwargs"},
        {"LHM_ARG_LOG_DISABLE",      "--log_disable"},
        {"LHM_ARG_LOG_FILE",         "--log_file"},
        {"LHM_ARG_VERBOSE",          "--verbose"},
        {"LHM_ARG_OFFLINE",          "--offline"},
    };
    return m;
}

// Boolean flags that are set via --flag / --no-flag pattern
static bool is_bool_flag(const std::string & key) {
    static const std::set<std::string> bool_keys = {
        "LHM_ARG_CONT_BATCHING",
        "LHM_ARG_MOE_OFFLOAD",
        "LHM_ARG_NO_MMAP",
        "LHM_ARG_MLOCK",
        "LHM_ARG_MODELS_AUTOLOAD",
        "LHM_ARG_JINJA",
        "LHM_ARG_NO_KV_OFFLOAD",
        "LHM_ARG_CACHE_PROMPT",
        "LHM_ARG_IGNORE_EOS",
        "LHM_ARG_METRICS",
        "LHM_ARG_SLOTS",
        "LHM_ARG_SIMPLE_IO",
        "LHM_ARG_NO_CONTEXT_SHIFT",
        "LHM_ARG_PREFILL_ASSISTANT",
        "LHM_ARG_SKIP_CHAT_PARSING",
        "LHM_ARG_WEBUI",
        "LHM_ARG_UI",
        "LHM_ARG_LOG_DISABLE",
        "LHM_ARG_VERBOSE",
        "LHM_ARG_OFFLINE",
    };
    return bool_keys.count(key) > 0;
}

std::vector<std::string> model_preset::to_args(const std::string & bin_path) const {
    std::vector<std::string> result;
    if (!bin_path.empty()) {
        result.push_back(bin_path);
    }
    const auto & flag_map = get_key_to_flag_map();
    for (const auto & [key, value] : options) {
        auto it = flag_map.find(key);
        if (it == flag_map.end()) {
            // Unknown key, skip (or could use --key style)
            continue;
        }
        const std::string & flag = it->second;
        if (is_bool_flag(key)) {
            bool is_true = (value == "1" || value == "true" || value == "on" || value == "yes" || value == "enabled");
            if (is_true) {
                result.push_back(flag);
            } else {
                // Use --no- variant
                result.push_back("--no-" + flag.substr(2));
            }
        } else {
            result.push_back(flag);
            result.push_back(value);
        }
    }
    return result;
}

void model_preset::apply_model_options(common_params & params, const std::set<std::string> & keys) const {
    for (const auto & [key, value] : options) {
        if (!keys.empty() && keys.find(key) == keys.end()) {
            continue;
        }
        if (key == "LHM_ARG_MODEL") {
            params.model.path = value;
        }
    }
}

// Convert model_preset options to a simple INI-like string for display
static std::string preset_to_ini_string(const model_preset & preset) {
    std::ostringstream ss;
    ss << "[" << preset.name << "]\n";
    for (const auto & [key, value] : preset.options) {
        ss << key << " = " << value << "\n";
    }
    ss << "\n";
    return ss.str();
}

// Helper: parse CLI args into a model_preset
static model_preset load_preset_from_args(int argc, char ** argv) {
    model_preset preset;
    preset.name = "default";
    const auto & flag_map = get_key_to_flag_map();
    // Build reverse map: flag -> key
    std::map<std::string, std::string> flag_to_key;
    for (const auto & [key, flag] : flag_map) {
        flag_to_key[flag] = key;
        // Also support --no- variants for bool flags
        if (is_bool_flag(key)) {
            flag_to_key["--no-" + flag.substr(2)] = key;
        }
    }
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        // Normalize underscores to dashes in flag names
        if (arg.compare(0, 2, "--") == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        auto it = flag_to_key.find(arg);
        if (it != flag_to_key.end()) {
            const std::string & key = it->second;
            if (is_bool_flag(key)) {
                // Check if it's a --no- variant
                bool is_neg = arg.compare(0, 5, "--no-") == 0;
                preset.set_option(key, is_neg ? "0" : "1");
            } else {
                // Value arg
                if (i + 1 < argc) {
                    preset.set_option(key, argv[++i]);
                }
            }
        }
        // Skip unknown args (they may be handled elsewhere)
    }
    return preset;
}

// Helper: parse INI file into presets
static model_presets load_presets_from_ini(const std::string & path, model_preset & global) {
    model_presets out;
    if (!std::filesystem::exists(path)) {
        return out;
    }

    std::ifstream file(path);
    if (!file.good()) {
        return out;
    }

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Simple INI parser
    std::string current_section;
    std::map<std::string, std::string> current_options;

    std::istringstream iss(contents);
    std::string line;
    while (std::getline(iss, line)) {
        // Trim whitespace
        auto trim = [](std::string & s) {
            size_t start = s.find_first_not_of(" \t\r\n");
            size_t end = s.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) { s.clear(); return; }
            s = s.substr(start, end - start + 1);
        };
        trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;

        if (line[0] == '[') {
            // Save previous section
            if (!current_section.empty() || !current_options.empty()) {
                model_preset preset;
                preset.name = current_section.empty() ? "default" : current_section;
                preset.options = current_options;
                if (preset.name == "*") {
                    global = preset;
                } else {
                    out[preset.name] = preset;
                }
            }
            // Parse new section
            auto end_bracket = line.find(']');
            if (end_bracket != std::string::npos) {
                current_section = line.substr(1, end_bracket - 1);
                auto trim2 = [](std::string & s) {
                    size_t start = s.find_first_not_of(" \t");
                    size_t end = s.find_last_not_of(" \t");
                    if (start == std::string::npos) { s.clear(); return; }
                    s = s.substr(start, end - start + 1);
                };
                trim2(current_section);
            }
            current_options.clear();
        } else {
            auto eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                auto trim2 = [](std::string & s) {
                    size_t start = s.find_first_not_of(" \t");
                    size_t end = s.find_last_not_of(" \t");
                    if (start == std::string::npos) { s.clear(); return; }
                    s = s.substr(start, end - start + 1);
                };
                trim2(key);
                trim2(value);
                if (!key.empty()) {
                    current_options[key] = value;
                }
            }
        }
    }
    // Save last section
    if (!current_section.empty() || !current_options.empty()) {
        model_preset preset;
        preset.name = current_section.empty() ? "default" : current_section;
        preset.options = current_options;
        if (preset.name == "*") {
            global = preset;
        } else {
            out[preset.name] = preset;
        }
    }
    return out;
}

// Helper: cascade (merge) global preset over a map of presets
static model_presets cascade_presets(const model_preset & global, const model_presets & base) {
    model_presets out;
    for (const auto & [name, preset] : base) {
        model_preset tmp = global;
        tmp.name = name;
        tmp.merge(preset);
        out[name] = std::move(tmp);
    }
    return out;
}

// Helper: check if a string value is truthy
static bool is_truthy_value(const std::string & val) {
    return val == "1" || val == "true" || val == "on" || val == "yes" || val == "enabled";
}

static void unset_reserved_args(model_preset & preset, bool unset_model_args) {
    preset.unset_option("LHM_ARG_SSL_KEY_FILE");
    preset.unset_option("LHM_ARG_SSL_CERT_FILE");
    preset.unset_option("LHM_API_KEY");
    preset.unset_option("LHM_ARG_MODELS_PRESET");
    if (unset_model_args) {
        preset.unset_option("LHM_ARG_MODEL");
        preset.unset_option("LHM_ARG_ALIAS");
    }
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) {
        return {};
    }

    const int len = static_cast<int>(std::wcslen(ws));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, ws, len, nullptr, 0, nullptr, nullptr);
    if (bytes == 0) {
        return {};
    }

    std::string utf8(bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, len, utf8.data(), bytes, nullptr, nullptr);

    return utf8;
}
#endif

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

void server_model_meta::update_args(std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option("LHM_ARG_HOST",  CHILD_ADDR);
    preset.set_option("LHM_ARG_PORT",  std::to_string(port));
    preset.set_option("LHM_ARG_ALIAS", name);
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LHM_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps() {
    // mtmd/multimodal support has been removed
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : base_params(params),
              base_env(get_environment()),
              base_preset(load_preset_from_args(argc, argv)) {
    // clean up base preset
    unset_reserved_args(base_preset, true);
    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        LOG_WARN("failed to get server executable path: %s\n", e.what());
        LOG_WARN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_models();
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LHM_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LHM_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    meta.update_args(bin_path); // render args
    meta.update_caps();
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<server_subproc>(),
        /* th      */ std::thread(),
        /* meta    */ std::move(meta)
    };
}

void server_models::notify_sse(const std::string & event, const std::string & model_id, const json & data) {
    std::unique_ptr<server_task_result_router> result = std::make_unique<server_task_result_router>();
    result->data = {
        {"model", model_id},
        {"event", event},
    };
    if (!data.is_null()) {
        result->data["data"] = data;
    }
    SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", event.c_str(), model_id.c_str(), safe_json_to_str(result->data).c_str());
    sse.broadcast(std::move(result));
}

void server_models::load_models() {

}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
    }
    return std::nullopt;
}

static int get_free_port() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return -1;
    }
    typedef SOCKET native_socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define CLOSE_SOCKET(s) closesocket(s)
#else
    typedef int native_socket_t;
#define INVALID_SOCKET_VAL -1
#define CLOSE_SOCKET(s) close(s)
#endif

    native_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VAL) {
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(0);

    if (bind(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

#ifdef _WIN32
    int namelen = sizeof(serv_addr);
#else
    socklen_t namelen = sizeof(serv_addr);
#endif
    if (getsockname(sock, (struct sockaddr*)&serv_addr, &namelen) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    int port = ntohs(serv_addr.sin_port);

    CLOSE_SOCKET(sock);
#ifdef _WIN32
    WSACleanup();
#endif

    return port;
}

// helper to convert vector<string> to char **
// pointers are only valid as long as the original vector is valid
static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec) {
    std::vector<char *> result;
    result.reserve(vec.size() + 1);
    for (const auto & s : vec) {
        result.push_back(const_cast<char*>(s.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

void server_models::load(const std::string & name) {
    if (!has_model(name)) {
        throw std::runtime_error("model name=" + name + " is not found");
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    auto meta = mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // prepare new instance info
    instance_t inst;
    inst.meta             = meta;
    inst.meta.port        = get_free_port();
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<server_subproc>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        inst.meta.update_args(bin_path); // render args

        std::vector<std::string> child_args = inst.meta.args; // copy
        std::vector<std::string> child_env  = base_env; // copy
        child_env.push_back("LHM_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        std::vector<char *> argv = to_char_ptr_array(child_args);
        std::vector<char *> envp = to_char_ptr_array(child_env);

        // TODO @ngxson : maybe separate stdout and stderr in the future
        //                so that we can use stdout for commands and stderr for logging
        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        inst.subproc->sproc.emplace();
        int result = subprocess_create_ex(argv.data(), options, envp.data(), nullptr, &inst.subproc->get());
        if (result != 0) {
            throw std::runtime_error("failed to spawn server instance");
        }

        inst.stdin_file = subprocess_stdin(&inst.subproc->get());
    }

    // start a thread to manage the child process
    // captured variables are guaranteed to be destroyed only after the thread is joined
    inst.th = std::thread([this, name, child_proc = inst.subproc, port = inst.meta.port, stop_timeout = inst.meta.stop_timeout]() {
        FILE * stdin_file = subprocess_stdin(&child_proc->get());
        FILE * stdout_file = subprocess_stdout(&child_proc->get()); // combined stdout/stderr

        std::thread log_thread([&]() {
            // read stdout/stderr and forward to main server log
            // also handle status report from child process
            std::vector<char> vec_buf(128 * 1024); // large buffer for storing info
            char * buffer = vec_buf.data();
            if (stdout_file) {
                while (fgets(buffer, vec_buf.size(), stdout_file) != nullptr) {
                    SRV_INF("[%5d] %s", port, buffer);
                    std::string str(buffer);
                    if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_READY)) {
                        this->update_status(name, SERVER_MODEL_STATUS_LOADED, 0);
                    } else if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_INFO)) {
                        this->update_loaded_info(name, str);
                    } else if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_SLEEP)) {
                        this->update_status(name, SERVER_MODEL_STATUS_SLEEPING, 0);
                    }
                }
            } else {
                SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            }
        });

        std::thread stopping_thread([&]() {
            // thread to monitor explicit stop requests; child crash is signalled via child_proc->stopped
            auto is_stopping = [this, &name]() {
                return this->stopping_models.find(name) != this->stopping_models.end();
            };
            {
                std::unique_lock<std::mutex> lk(this->mutex);
                this->cv_stop.wait(lk, [&]() {
                    return is_stopping() || child_proc->stopped.load(std::memory_order_acquire);
                });
            }
            // child crashed or finished on its own — skip graceful shutdown sequence
            if (child_proc->stopped.load(std::memory_order_acquire)) {
                return;
            }
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
            fflush(stdin_file);
            int64_t start_time = ggml_time_ms();
            while (true) {
                std::unique_lock<std::mutex> lk(this->mutex);
                if (!is_stopping() || child_proc->stopped.load(std::memory_order_acquire)) {
                    return;
                }
                int64_t elapsed = ggml_time_ms() - start_time;
                if (elapsed >= stop_timeout * 1000) {
                    lk.unlock();
                    SRV_WRN("force-killing model instance name=%s after %d seconds timeout\n", name.c_str(), stop_timeout);
                    child_proc->terminate();
                    return;
                }
                this->cv_stop.wait_for(lk, std::chrono::seconds(1), [&]() {
                    return !is_stopping() || child_proc->stopped.load(std::memory_order_acquire);
                });
            }
        });

        // we reach here when the child process exits (stdout EOF)
        // note: we cannot join() prior to this point because it will close stdin_file
        if (log_thread.joinable()) {
            log_thread.join();
        }

        child_proc->stopped.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(this->mutex);
            stopping_models.erase(name);
            cv_stop.notify_all();
        }
        if (stopping_thread.joinable()) {
            stopping_thread.join();
        }

        // get the exit code
        int exit_code = 0;
        subprocess_join(&child_proc->get(), &exit_code);
        subprocess_destroy(&child_proc->get());

        // update status and exit code
        this->update_status(name, SERVER_MODEL_STATUS_UNLOADED, exit_code);
        SRV_INF("instance name=%s exited with status %d\n", name.c_str(), exit_code);
    });

    // clean up old process/thread if exists
    {
        auto & old_instance = mapping[name];
        // old process should have exited already, but just in case, we clean it up here
        if (old_instance.subproc->is_alive()) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            old_instance.subproc->terminate(); // force kill
        }
        if (old_instance.th.joinable()) {
            old_instance.th.join();
        }
    }

    notify_sse("model_status", name, {
        {"status", server_model_status_to_string(inst.meta.status)},
    });

    mapping[name] = std::move(inst);
    cv.notify_all();
}

void server_models::unload(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            stopping_models.insert(name);
            if (it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
                // special case: if model is in loading state, unloading means force-killing it
                SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
                it->second.subproc->terminate();
            }
            cv_stop.notify_all();
            // status change will be handled by the managing thread
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lk(mutex);
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) {
                SRV_INF("stopping model instance name=%s\n", name.c_str());
                stopping_models.insert(name);
                cv_stop.notify_all();
                // status change will be handled by the managing thread
            }
            // moving the thread to join list to avoid deadlock
            to_join.push_back(std::move(inst.th));
        }
    }
    for (auto & th : to_join) {
        if (th.joinable()) {
            th.join();
        }
    }
}

void server_models::update_status(const std::string & name, server_model_status status, int exit_code) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.status    = status;
        meta.exit_code = exit_code;
    }
    // broadcast status change to SSE
    {
        json data = {
            {"status", server_model_status_to_string(status)},
        };
        if (status == SERVER_MODEL_STATUS_UNLOADED) {
            data["exit_code"] = exit_code;
        }
        // note: notify_sse doesn't acquire the lock, so no deadlock here
        notify_sse("status_change", name, data);
    }
    cv.notify_all();
}

void server_models::update_loaded_info(const std::string & name, std::string & raw_info) {
    if (!string_starts_with(raw_info, CMD_CHILD_TO_ROUTER_INFO)) {
        SRV_WRN("invalid loaded info format from child for model name=%s: %s\n", name.c_str(), raw_info.c_str());
        return;
    }

    json info;
    try {
        info = json::parse(raw_info.substr(strlen(CMD_CHILD_TO_ROUTER_INFO)));
    } catch (const std::exception & e) {
        SRV_WRN("failed to parse loaded info from child for model name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.loaded_info = info;
    }
    cv.notify_all();
}

bool server_models::remove(const std::string & name) {
    auto meta = get_meta(name);

    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }

    unload(name); // stop running instance
    {
        std::unique_lock<std::mutex> lk(mutex);
        // a stopped instance lands on UNLOADED
        wait(lk, name, [](const server_model_meta & new_meta) {
            return new_meta.status == SERVER_MODEL_STATUS_UNLOADED;
        });
        // join before erasing - after status reaches UNLOADED the thread no
        // longer acquires this mutex, so joining while holding it is safe
        if (mapping[name].th.joinable()) {
            mapping[name].th.join();
        }
        mapping.erase(name);
        SRV_INF("removing model name=%s from list\n", name.c_str());
        notify_sse("model_remove", name, {});
        return true;
    }
}

void server_models::wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    std::unique_lock<std::mutex> lk(mutex);
    wait(lk, name, predicate);
}

void server_models::wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    cv.wait(lk, [this, &name, &predicate]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return predicate(it->second.meta);

        }
        return false;
    });
}

bool server_models::ensure_model_ready(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->is_ready()) {
        return false; // ready for taking requests
    }
    if (meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // child is sleeping but still running; new request will wake it up
    }
    if (meta->status == SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model name=%s is not loaded, loading...\n", name.c_str());
        load(name);
    }

    // wait for loading to complete
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    wait(name, [&meta](const server_model_meta & new_meta) {
        if (new_meta.status != SERVER_MODEL_STATUS_LOADING) {
            meta = new_meta; // update meta for final check after wait
            return true;
        }
        return false;
    });

    // check final status
    if (!meta.has_value() || meta->is_failed()) {
        throw std::runtime_error("model name=" + name + " failed to load");
    }

    return true;
}

bool server_models::is_child_server() {
    const char * router_port = std::getenv("LHM_SERVER_ROUTER_PORT");
    return router_port != nullptr;
}

std::thread server_models::setup_child_server(const std::function<void(int)> & shutdown_handler, const json & model_info) {
    // send a notification to the router server that a model instance is ready
    fflush(stdout);
    fprintf(stdout, "%s\n", CMD_CHILD_TO_ROUTER_READY);
    fflush(stdout);
    fprintf(stdout, "%s%s\n", CMD_CHILD_TO_ROUTER_INFO, safe_json_to_str(model_info).c_str());
    fflush(stdout);

    // setup thread for monitoring stdin
    return std::thread([shutdown_handler]() {
        // wait for EOF on stdin
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF detected, that means the router server is unexpectedly exit or killed
                eof = true;
                break;
            }
            if (line.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);
        }
    });
}

void server_models::notify_router_sleeping_state(bool is_sleeping) {
    fflush(stdout);
    fprintf(stdout, "%s\n", is_sleeping ? CMD_CHILD_TO_ROUTER_SLEEP : CMD_CHILD_TO_ROUTER_READY);
    fflush(stdout);
}


// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> sse_client_id_counter = 0;
struct server_models_sse_client {
    server_response & queue_results;
    int client_id;
    server_models_sse_client(server_response & q)
            : queue_results(q), client_id(sse_client_id_counter.fetch_add(1, std::memory_order_relaxed)) {
        SRV_DBG("new SSE client connected, assigned client_id=%d\n", client_id);
        queue_results.add_waiting_task_id(client_id);
    }
    ~server_models_sse_client() {
        SRV_DBG("SSE client disconnected, removing client_id=%d\n", client_id);
        queue_results.remove_waiting_task_id(client_id);
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            static const int http_polling_seconds = 1; // check should_stop every 1 second
            server_task_result_ptr result = queue_results.recv_with_timeout({client_id}, http_polling_seconds);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    return nullptr;
                }
                // continue waiting otherwise
            } else {
                SRV_DBG("recv result for client_id=%d: %s\n", client_id, safe_json_to_str(result->to_json()).c_str());
                return result;
            }
        }
        // should not reach here
    }
};

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};
    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop()) {
                close_read(); // signal broken pipe to writer
                return false; // cancelled / reader no longer alive
            }
            cv.wait_for(lk, poll_interval);
        }
    }
    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static std::string generate_multipart_boundary() {
    thread_local std::mt19937 gen(std::random_device{}());
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string boundary = "----lhm_cpp-proxy-";
    for (int i = 0; i < 16; i++) {
        boundary += chars[dis(gen)];
    }
    return boundary;
}

static std::string build_multipart_body(
        const json & form_fields,
        const std::map<std::string, uploaded_file> & files,
        const std::string & boundary) {
    static auto sanitize_field = [](const std::string & text) {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    };

    std::ostringstream body;

    for (const auto & [key, value] : form_fields.items()) {
        if (value.is_array()) {
            for (const auto & item : value) {
                body << "--" << boundary << "\r\n";
                body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
                body << "\r\n";
                if (!item.is_string()) {
                    throw std::invalid_argument("expected string");
                }
                body << item.get<std::string>() << "\r\n";
            }
        } else {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
            body << "\r\n";
            if (!value.is_string()) {
                throw std::invalid_argument("expected string");
            }
            body << value.get<std::string>() << "\r\n";
        }
    }

    for (const auto & [key, file] : files) {
        body << "--" << boundary << "\r\n";
        body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"";
        if (!file.filename.empty()) {
            body << "; filename=\"" << sanitize_field(file.filename) << "\"";
        }
        body << "\r\n";
        if (!file.content_type.empty()) {
            body << "Content-Type: " << sanitize_field(file.content_type) << "\r\n";
        } else {
            body << "Content-Type: application/octet-stream\r\n";
        }
        body << "\r\n";
        body.write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        body << "\r\n";
    }

    body << "--" << boundary << "--\r\n";
    return body.str();
}
