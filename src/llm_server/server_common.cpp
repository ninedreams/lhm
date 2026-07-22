#include "common.h"
#include "log.h"
#include "lhm.h"
#include "chat/chat.h"
#include "base64.hpp"


#include "server_common.h"

#include <random>
#include <sstream>
#include <fstream>

json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
        case ERROR_TYPE_EXCEED_CONTEXT_SIZE:
            type_str = "exceed_context_size_error";
            code = 400;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

//
// random string / id
//

std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

std::string gen_tool_call_id() {
    return random_string();
}

const char * get_media_marker() {
    static const std::string marker = []() {
        // allow user to pin a reproducible marker via env var
        const char * env = getenv("LHM_MEDIA_MARKER");
        if (env && env[0] != '\0') {
            return std::string(env);
        }
        return std::string("<__media_") + random_string() + "__>";
    }();
    return marker.c_str();
}

//
// lora utils
//

bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras) {
    bool found_alora = false;
    for (const auto & lora : loras) {
        if (lora.scale != 0) {
            if (lhm_adapter_get_alora_n_invocation_tokens(lora.ptr) == 0) {
                return false;
            }
            found_alora = true;
        }
    }
    return found_alora;
}

bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next) {

    // This should always be called after determining that the two sets are
    // _not_ equal. This assert is therefore some slightly wasted work and
    // should be safe to remove as long as this method is called correctly.
    LHM_ASSERT(!are_lora_equal(current, next));

    return (
        !(lora_get_enabled_ids(current).empty() || lora_all_alora(current)) ||
        !lora_all_alora(next));
}

std::map<int, float> parse_lora_request(const json & data) {
    std::map<int, float> lora;

    // set value
    for (const auto & entry : data) {
        int id      = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        lora[id] = scale;
    }

    return lora;
}

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2) {
    if (l1.size() != l2.size()) {
        return false;
    }
    for (size_t i = 0; i < l1.size(); ++i) {
        // we don't check lora.path to reduce the time complexity
        if (l1[i].scale != l2[i].scale || l1[i].ptr != l2[i].ptr) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras) {
    std::vector<size_t> enabled_ids;
    for (size_t i = 0; i < loras.size(); ++i) {
        if (loras[i].scale > 0) {
            enabled_ids.push_back(i);
        }
    }
    return enabled_ids;
}

//
// base64 utils (TODO: use the lhm::base64::decode from base64.hpp)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline raw_buffer base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    raw_buffer ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// server_tokens implementation
//
// server_tokens implementation
//

server_tokens::server_tokens(const lhm_tokens & tokens) : tokens(tokens) {
}

lhm_pos server_tokens::pos_next(int64_t n_tokens) const {
    if (n_tokens < 0) {
        return tokens.size();
    }

    return n_tokens;
}

size_t server_tokens::size_up_to_pos(lhm_pos max_pos) const {
    return std::min((size_t)max_pos, tokens.size());
}

std::string server_tokens::str() const {
    std::ostringstream oss;
    oss << "tokens: ";
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        oss << "idx:" << idx << " " << tokens[idx] << " ";
    }
    return oss.str();
}

void server_tokens::push_back(lhm_token tok) {
    tokens.emplace_back(tok);
}

void server_tokens::push_back(server_tokens & tokens) {
    for (size_t i = 0; i < tokens.size(); i++) {
        push_back(tokens[i]);
    }
}

void server_tokens::insert(const lhm_tokens & inp_tokens) {
    tokens.insert(tokens.end(), inp_tokens.begin(), inp_tokens.end());
}

const lhm_tokens & server_tokens::get_tokens() const {
    return tokens;
}

lhm_tokens server_tokens::get_text_tokens() const {
    return tokens;
}

void server_tokens::set_token(lhm_pos pos, lhm_token id) {
    tokens[pos] = id;
}

void server_tokens::keep_first(size_t n) {
    LHM_ASSERT(n <= tokens.size());
    tokens.resize(n);
}

std::string server_tokens::detokenize(const lhm_context * ctx, bool special) const {
    return common_detokenize(ctx, tokens, special);
}

size_t server_tokens::get_common_prefix(const server_tokens & b) const {
    const size_t max_idx = std::min(tokens.size(), b.tokens.size());

    for (size_t i = 0; i < max_idx; ++i) {
        if (tokens[i] == b.tokens[i]) {
            continue;
        }

        return i;
    }

    return max_idx;
}

bool server_tokens::validate(const struct lhm_context * ctx) const {
    const lhm_model * model = lhm_get_model(ctx);
    const lhm_vocab * vocab = lhm_model_get_vocab(model);
    const int32_t n_vocab = lhm_vocab_n_tokens(vocab);

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto & t = tokens[i];
        if (t < 0 || t >= n_vocab) {
            return false;
        }
    }
    return true;
}

server_tokens server_tokens::clone() const {
    server_tokens res;
    res.tokens = tokens;
    return res;
}
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool json_is_array_of_mixed_numbers_strings(const json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

bool json_is_array_and_contains_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (e.is_number_integer()) {
                return true;
            }
        }
        return false;
    }
    return false;
}

json json_get_nested_values(const std::vector<std::string> & paths, const json & js) {
    json result = json::object();

    for (const std::string & path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string & k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}

lhm_tokens tokenize_mixed(const lhm_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    lhm_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                lhm_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<lhm_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string" }
 */
static server_tokens tokenize_input_subprompt(const lhm_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        lhm_tokens tmp = tokenize_mixed(vocab, json_prompt, add_special, parse_special);
        return server_tokens(tmp);
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        lhm_tokens tmp = json_prompt.get<lhm_tokens>();
        return server_tokens(tmp);
    } else if (json_prompt.contains(JSON_STRING_PROMPT_KEY)) {
        // JSON object with prompt key.
        lhm_tokens tmp = tokenize_mixed(vocab, json_prompt.at(JSON_STRING_PROMPT_KEY), add_special, parse_special);
        return server_tokens(tmp);
   } else {
       throw std::runtime_error("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
   }
}

std::vector<server_tokens> tokenize_input_prompts(const lhm_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    std::vector<server_tokens> result;
    if (json_prompt.is_array() && !json_is_array_and_contains_numbers(json_prompt)) {
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            result.push_back(tokenize_input_subprompt(vocab, p, add_special, parse_special));
        }
    } else {
        result.push_back(tokenize_input_subprompt(vocab, json_prompt, add_special, parse_special));
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

//
// OAI utils
//

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body) {
    json lhm_params;

    if (!body.contains("prompt")) {
        throw std::runtime_error("\"prompt\" is required");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        lhm_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        lhm_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "echo" field
    if (json_value(body, "echo", false)) {
        throw std::runtime_error("Only no echo is supported");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params { "best_of", "suffix" };
    for (const auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to lhm_params
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!lhm_params.contains(item.key()) || item.key() == "n_predict") {
            lhm_params[item.key()] = item.value();
        }
    }

    return lhm_params;
}

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files)
{
    json lhm_params;

    auto tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    auto tool_choice = json_value(body, "tool_choice", std::string("auto"));

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        lhm_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        lhm_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            if (response_format.contains("schema") || json_schema.empty()) {
                json_schema = json_value(response_format, "schema", json::object());
            }
        } else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::invalid_argument("'messages' is required");
    }
    json & messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::invalid_argument("Expected 'messages' to be an array");
    }
    for (auto & msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::invalid_argument("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json & content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::invalid_argument("Expected 'content' to be a string or an array");
        }

        for (auto & p : content) {
            std::string type      = json_value(p, "type", std::string());
            if (type != "text") {
                throw std::invalid_argument("unsupported content[].type");
            }
        }
    }

    auto caps = common_chat_templates_get_caps(opt.tmpls.get());

    common_chat_templates_inputs inputs;
    inputs.messages               = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                  = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice            = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema            = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar                = grammar;
    inputs.use_jinja              = opt.use_jinja;
    inputs.parallel_tool_calls    = json_value(body, "parallel_tool_calls", caps["supports_parallel_tool_calls"]);
    inputs.add_generation_prompt  = json_value(body, "add_generation_prompt", true);
    inputs.continue_final_message = body.contains("continue_final_message") ?
        common_chat_continuation_parse(body.at("continue_final_message")) :
        COMMON_CHAT_CONTINUATION_NONE;
    if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_NONE && opt.prefill_assistant
        && !inputs.messages.empty() && inputs.messages.back().role == "assistant") {
        if (inputs.messages.size() >= 2 && inputs.messages[inputs.messages.size() - 2].role == "assistant") {
            throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
        }
        inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
        inputs.add_generation_prompt  = false;
    }
    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && inputs.add_generation_prompt) {
        throw std::invalid_argument("Cannot set both add_generation_prompt and continue_final_message to true.");
    }
    inputs.reasoning_format = opt.reasoning_format;
    if (body.contains("reasoning_format")) {
        inputs.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
    }
    inputs.enable_thinking = opt.enable_thinking;
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
        }
        lhm_params["parse_tool_calls"] = true;
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto & item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    } else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    } else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    inputs.force_pure_content = opt.force_pure_content;

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls.get(), inputs);

    lhm_params["chat_format"] = static_cast<int>(chat_params.format);
    lhm_params["prompt"]      = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        lhm_params["grammar"]      = chat_params.grammar;
        lhm_params["grammar_type"] = std::string("tool_calls");
    }
    lhm_params["grammar_lazy"] = chat_params.grammar_lazy;
    auto grammar_triggers        = json::array();
    for (const auto & trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    lhm_params["grammar_triggers"]  = grammar_triggers;
    lhm_params["preserved_tokens"]  = chat_params.preserved_tokens;
    lhm_params["generation_prompt"] = chat_params.generation_prompt;
    for (const auto & stop : chat_params.additional_stops) {
        lhm_params["stop"].push_back(stop);
    }
    if (!chat_params.parser.empty()) {
        lhm_params["chat_parser"] = chat_params.parser;
    }

    lhm_params["message_spans"] = json::array();

    for (const auto & span : chat_params.message_spans) {
        lhm_params["message_spans"].push_back({
            { "role", span.role },
            { "pos",  span.pos  },
            { "len",  span.len  },
        });
    }

    // Reasoning budget: pass parameters through to sampling layer
    {
        int reasoning_budget = json_value(body, "thinking_budget_tokens", -1);
        if (reasoning_budget == -1) {
            reasoning_budget = opt.reasoning_budget;
        }

        if (!chat_params.thinking_end_tag.empty()) {
            lhm_params["reasoning_budget_tokens"] = reasoning_budget;
            lhm_params["reasoning_budget_start_tag"] = chat_params.thinking_start_tag;
            lhm_params["reasoning_budget_end_tag"] = chat_params.thinking_end_tag;
            lhm_params["reasoning_budget_message"] = opt.reasoning_budget_message;
            lhm_params["reasoning_control"] = json_value(body, "reasoning_control", false);
        }
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::invalid_argument("logprobs is not supported with tools + stream");
        }
        lhm_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to lhm_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!lhm_params.contains(item.key()) || item.key() == "n_predict") {
            lhm_params[item.key()] = item.value();
        }
    }

    return lhm_params;
}

json format_embeddings_response_oaicompat(
        const json & request,
        const std::string & model_name,
        const json & embeddings,
        bool use_base64) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto & elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", lhm::base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        } else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);

        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }

    json res = json {
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n) {
    int32_t n_tokens = 0;
    bool return_text = is_tei_format && json_value(request, "return_text", false);
    std::vector<json> elements; // Temporary vector to hold unsorted elements
    std::string score_label = is_tei_format ? "score" : "relevance_score";
    for (const auto & rank : ranks) {
        int index = json_value(rank, "index", 0);
        json elem = json{
            {"index", index},
            {score_label, json_value(rank, "score", 0.0)},
        };
        n_tokens += json_value(rank, "tokens_evaluated", 0);
        if (return_text) {
            elem["text"] = std::move(texts[index]);
        }
        elements.push_back(elem);
    }

    std::sort(elements.begin(), elements.end(), [score_label](const json& a, const json& b) {
        return json_value(a, score_label, 0.0) > json_value(b, score_label, 0.0);
    });

    elements.resize(std::min(top_n, (int)elements.size()));
    json results = elements;

    if (is_tei_format) return results;

    json res = json{
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json{
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"results", results}
    };

    return res;
}


//
// other utils
//

std::vector<lhm_token_data> get_token_probabilities(lhm_context * ctx, int idx, size_t n_top) {
    std::vector<lhm_token_data> cur;

    const auto * logits = lhm_get_logits_ith(ctx, idx);
    const lhm_token * sampled_ids = lhm_get_sampled_candidates_ith(ctx, idx);

    const int n_logits = lhm_get_sampled_logits_count_ith(ctx, idx);

    cur.resize(n_logits);
    if (sampled_ids) {
        for (int i = 0; i < n_logits; i++) {
            cur[i] = lhm_token_data{sampled_ids[i], logits[i], 0.0f};
        }
    } else {
        for (lhm_token token_id = 0; token_id < n_logits; token_id++) {
            cur[token_id] = lhm_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    // sort tokens by logits (partial: only the leading `n_top` need ordering)
    if (n_top > cur.size()) {
        n_top = cur.size();
    }
    if (n_top > 0) {
        std::partial_sort(cur.begin(), cur.begin() + n_top, cur.end(),
            [](const lhm_token_data & a, const lhm_token_data & b) {
                return a.logit > b.logit;
            });
    }

    // apply softmax
    float max_l = -std::numeric_limits<float>::infinity();
    if (n_top > 0) {
        max_l = cur[0].logit; // partial_sort guarantees the absolute maximum is at index 0
    } else {
        for (const auto & t : cur) {
            max_l = std::max(max_l, t.logit);
        }
    }
    float cum_sum = 0.0f;
    for (auto & t : cur) {
        float p = expf(t.logit - max_l);
        t.p = p;
        cum_sum += p;
    }
    for (auto & t : cur) {
        t.p /= cum_sum;
    }

    return cur;
}

std::string safe_json_to_str(const json & data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}

// TODO: reuse lhm_detokenize
template <class Iter>
static std::string tokens_to_str(const lhm_vocab * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

std::string tokens_to_str(lhm_context * ctx, const lhm_tokens & tokens) {
    auto model = lhm_get_model(ctx);
    return tokens_to_str(lhm_model_get_vocab(model), tokens.begin(), tokens.end());
}

std::string tokens_to_str(const lhm_vocab * vocab, const lhm_tokens & tokens) {
    return tokens_to_str(vocab, tokens.begin(), tokens.end());
}

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const lhm_context * ctx, const lhm_token token) {
    std::string out = token == LHM_TOKEN_NULL ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

// format server_sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & data) {
        ss << "data: " <<
            safe_json_to_str(data) <<
            "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_oai_resp_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & event_obj) {
        ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
        ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_anthropic_sse(const json & data) {
    std::ostringstream ss;

    auto send_event = [&ss](const json & event_obj) {
        if (event_obj.contains("event") && event_obj.contains("data")) {
            ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
            ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
        } else {
            ss << "data: " << safe_json_to_str(event_obj) << "\n\n";
        }
    };

    if (data.is_array()) {
        for (const auto & event : data) {
            send_event(event);
        }
    } else {
        send_event(data);
    }

    return ss.str();
}

bool is_valid_utf8(const std::string & str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    const unsigned char* end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 ||
                (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

lhm_tokens format_prompt_infill(
        const lhm_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const lhm_tokens & tokens_prompt
    ) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    lhm_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (lhm_vocab_fim_rep(vocab) != LHM_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject", false, false);

        extra_tokens.push_back(lhm_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text",     std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (lhm_vocab_fim_sep(vocab) != LHM_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "", false, false);

            extra_tokens.insert(extra_tokens.end(), lhm_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (lhm_vocab_fim_sep(vocab) != LHM_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename", false, false);

        extra_tokens.insert(extra_tokens.end(), lhm_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(),                3*(n_batch/4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch/4) - (2 + tokens_prompt.size())));

    LOG_DEBUG("n_prefix_take = {}, n_suffix_take = {}, total = {}", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) - 2*n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), lhm_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(),   tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), lhm_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (lhm_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), lhm_vocab_bos(vocab));
    }

    LOG_DEBUG("extra: n_ctx = {}, n_extra_take = {}, n_extra = {}", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(lhm_vocab_fim_mid(vocab));

    return embd_inp;
}

server_tokens format_prompt_rerank(
        const struct lhm_model * model,
        const struct lhm_vocab * vocab,
        const std::string & query,
        const std::string & doc) {
    server_tokens result = {};

    const char * rerank_prompt = lhm_model_chat_template(model, "rerank");

    if (rerank_prompt != nullptr) {
        std::string prompt = rerank_prompt;
        string_replace_all(prompt, "{query}"   , query);
        string_replace_all(prompt, "{document}", doc  );
        server_tokens tokens = tokenize_input_subprompt(vocab, prompt, false, true);
        result.push_back(tokens);
    } else {
        // Get EOS token - use SEP token as fallback if EOS is not available
        server_tokens query_tokens = tokenize_input_subprompt(vocab, query, false, false);
        server_tokens doc_tokens   = tokenize_input_subprompt(vocab, doc,   false, false);
        lhm_token eos_token = lhm_vocab_eos(vocab);
        if (eos_token == LHM_TOKEN_NULL) {
            eos_token = lhm_vocab_sep(vocab);
        }

        if (lhm_vocab_get_add_bos(vocab)) {
            result.push_back(lhm_vocab_bos(vocab));
        }
        result.push_back(query_tokens);
        if (lhm_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
        if (lhm_vocab_get_add_sep(vocab)) {
            result.push_back(lhm_vocab_sep(vocab));
        }
        result.push_back(doc_tokens);
        if (lhm_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
    }

    return result;
}
