#pragma once

#include "lhm.h"

#include <string>
#include <vector>
#include <memory>

// pre-tokenization types
enum lhm_vocab_pre_type {
    LHM_VOCAB_PRE_TYPE_DEFAULT           = 0,
    LHM_VOCAB_PRE_TYPE_QWEN35            = 1
};

struct LLM_KV;
struct lhm_model_loader;

struct lhm_vocab {
    struct token_data {
        std::string      text;
        float            score;
        lhm_token_attr attr;
    };

    struct normalizer_options {
        bool lowercase     = true;
        bool strip_accents = true;
        // TODO: clean_text, handle_chinese_chars
    };

    lhm_vocab();
    ~lhm_vocab();

    void load(lhm_model_loader & ml, const LLM_KV & kv);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum lhm_vocab_type     get_type()     const;
    enum lhm_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (lhm_token id) const;
    bool is_unknown     (lhm_token id) const;
    bool is_control     (lhm_token id) const;
    bool is_byte        (lhm_token id) const;
    bool is_user_defined(lhm_token id) const;
    bool is_unused      (lhm_token id) const;
    bool is_eog         (lhm_token id) const;

    uint8_t     token_to_byte(lhm_token id) const;
    lhm_token byte_to_token(uint8_t ch)     const;

    lhm_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(lhm_token id) const;

    const char *     token_get_text (lhm_token id) const;
    float            token_get_score(lhm_token id) const;
    lhm_token_attr token_get_attr (lhm_token id) const;

    lhm_token token_bos() const;
    lhm_token token_eos() const;
    lhm_token token_eot() const;
    lhm_token token_eom() const;
    lhm_token token_unk() const;
    lhm_token token_sep() const;
    lhm_token token_nl () const;
    lhm_token token_pad() const;
    lhm_token token_mask() const;

    lhm_token token_prefix() const;
    lhm_token token_middle() const;
    lhm_token token_suffix() const;

    lhm_token token_fim_pre() const;
    lhm_token token_fim_suf() const;
    lhm_token token_fim_mid() const;
    lhm_token token_fim_pad() const;
    lhm_token token_fim_rep() const;
    lhm_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;
    const normalizer_options & get_normalizer_opts() const;

    const std::vector<lhm_token> & get_suppress_tokens() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  lhm_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<lhm_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  lhm_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(lhm_token token) const;

    int32_t detokenize(
            const lhm_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<lhm_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
