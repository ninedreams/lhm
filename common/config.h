#pragma once

#include <gflags/gflags.h>

namespace lhm {

void init_config(int argc, char ** argv);

}

// ============================================================================
// Log parameters (originally in config.h)
// ============================================================================
DECLARE_string(log_file);
DECLARE_int32(log_rotate_hour);
DECLARE_int32(log_rotate_minute);
DECLARE_string(log_level);
DECLARE_string(log_pattern);

// ============================================================================
// General parameters
// ============================================================================
DECLARE_bool(verbose_prompt);
DECLARE_bool(display_prompt);
DECLARE_string(color);
DECLARE_int32(threads);
DECLARE_int32(threads_batch);
DECLARE_string(cpu_mask);
DECLARE_string(cpu_range);
DECLARE_string(cpu_strict);
DECLARE_int32(prio);
DECLARE_string(poll);
DECLARE_string(cpu_mask_batch);
DECLARE_string(cpu_range_batch);
DECLARE_int32(cpu_strict_batch);
DECLARE_int32(prio_batch);
DECLARE_int32(poll_batch);

// ============================================================================
// Context / batch parameters
// ============================================================================
DECLARE_int32(ctx_size);
DECLARE_int32(predict);
DECLARE_int32(batch_size);
DECLARE_int32(ubatch_size);
DECLARE_int32(keep);
DECLARE_bool(swa_full);
DECLARE_int32(ctx_checkpoints);
DECLARE_int32(checkpoint_min_step);
DECLARE_int32(cache_ram);
DECLARE_bool(kv_unified);
DECLARE_bool(cache_idle_slots);
DECLARE_bool(context_shift);
DECLARE_int32(chunks);

// ============================================================================
// Flash attention
// ============================================================================
DECLARE_string(flash_attn);

// ============================================================================
// Prompt parameters
// ============================================================================
DECLARE_string(prompt);
DECLARE_string(system_prompt);
DECLARE_bool(perf);
DECLARE_bool(show_timings);
DECLARE_string(file);
DECLARE_string(system_prompt_file);
DECLARE_string(in_file);
DECLARE_string(binary_file);
DECLARE_bool(escape);
DECLARE_int32(print_token_count);
DECLARE_string(prompt_cache);
DECLARE_bool(prompt_cache_all);
DECLARE_bool(prompt_cache_ro);
DECLARE_string(reverse_prompt);
DECLARE_bool(special);
DECLARE_bool(conversation);
DECLARE_bool(single_turn);
DECLARE_bool(interactive);
DECLARE_bool(interactive_first);
DECLARE_bool(multiline_input);
DECLARE_bool(in_prefix_bos);
DECLARE_string(in_prefix);
DECLARE_string(in_suffix);
DECLARE_bool(warmup);
DECLARE_bool(spm_infill);

// ============================================================================
// Sampling parameters
// ============================================================================
DECLARE_string(samplers);
DECLARE_string(seed);
DECLARE_string(sampler_seq);
DECLARE_bool(ignore_eos);
DECLARE_string(temp);
DECLARE_int32(top_k);
DECLARE_string(top_p);
DECLARE_string(min_p);
DECLARE_string(top_nsigma);
DECLARE_string(xtc_probability);
DECLARE_string(xtc_threshold);
DECLARE_string(typical);
DECLARE_int32(repeat_last_n);
DECLARE_string(repeat_penalty);
DECLARE_string(presence_penalty);
DECLARE_string(frequency_penalty);
DECLARE_string(dry_multiplier);
DECLARE_string(dry_base);
DECLARE_int32(dry_allowed_length);
DECLARE_int32(dry_penalty_last_n);
DECLARE_string(dry_sequence_breaker);
DECLARE_string(adaptive_target);
DECLARE_string(adaptive_decay);
DECLARE_string(dynatemp_range);
DECLARE_string(dynatemp_exp);
DECLARE_int32(mirostat);
DECLARE_string(mirostat_lr);
DECLARE_string(mirostat_ent);
DECLARE_string(logit_bias);
DECLARE_string(grammar);
DECLARE_string(grammar_file);
DECLARE_string(json_schema);
DECLARE_string(json_schema_file);
DECLARE_bool(backend_sampling);

// ============================================================================
// Model type parameters
// ============================================================================
DECLARE_string(pooling);
DECLARE_string(attention);
DECLARE_string(rope_scaling);
DECLARE_string(rope_scale);
DECLARE_string(rope_freq_base);
DECLARE_string(rope_freq_scale);
DECLARE_int32(yarn_orig_ctx);
DECLARE_string(yarn_ext_factor);
DECLARE_string(yarn_attn_factor);
DECLARE_string(yarn_beta_slow);
DECLARE_string(yarn_beta_fast);
DECLARE_int32(grp_attn_n);
DECLARE_int32(grp_attn_w);

// ============================================================================
// KV cache / offload parameters
// ============================================================================
DECLARE_bool(kv_offload);
DECLARE_bool(repack);
DECLARE_bool(no_host);
DECLARE_string(cache_type_k);
DECLARE_string(cache_type_v);
DECLARE_string(defrag_thold);

// ============================================================================
// Use mooncake to KV cache / offload parameters
// ============================================================================
DECLARE_bool(enable_mooncake);
DECLARE_string(mooncake_protocol);
DECLARE_string(mooncake_engine_meta_url);
DECLARE_string(mooncake_master_server_entry);
DECLARE_string(mooncake_device_name);

// ============================================================================
// Parallel / batching
// ============================================================================
DECLARE_int32(parallel);
DECLARE_int32(sequences);
DECLARE_bool(cont_batching);

// ============================================================================
// RPC / memory parameters
// ============================================================================
DECLARE_string(rpc);
DECLARE_bool(mlock);
DECLARE_bool(mmap);
DECLARE_bool(direct_io);
DECLARE_string(numa);
DECLARE_string(device);
DECLARE_bool(list_devices);
DECLARE_string(override_tensor);
DECLARE_bool(cpu_moe);
DECLARE_int32(n_cpu_moe);

// ============================================================================
// GPU parameters
// ============================================================================
DECLARE_string(gpu_layers);
DECLARE_string(split_mode);
DECLARE_string(tensor_split);
DECLARE_int32(main_gpu);

// ============================================================================
// Fit parameters
// ============================================================================
DECLARE_string(fit);
DECLARE_string(fit_print);
DECLARE_string(fit_target);
DECLARE_int32(fit_ctx);
DECLARE_bool(check_tensors);

// ============================================================================
// Override / LoRA / control vector parameters
// ============================================================================
DECLARE_string(override_kv);
DECLARE_bool(op_offload);
DECLARE_string(lora);
DECLARE_string(lora_scaled);
DECLARE_string(control_vector);
DECLARE_string(control_vector_scaled);
DECLARE_string(control_vector_layer_range);
DECLARE_bool(lora_init_without_apply);

// ============================================================================
// Model path parameters
// ============================================================================
DECLARE_string(alias);
DECLARE_string(tags);
DECLARE_string(model);

// ============================================================================
// Retrieval / embedding parameters
// ============================================================================
DECLARE_string(context_file);
DECLARE_int32(chunk_size);
DECLARE_string(chunk_separator);
DECLARE_int32(embd_normalize);
DECLARE_string(embd_output_format);
DECLARE_string(embd_separator);
DECLARE_string(cls_separator);
DECLARE_bool(embedding);
DECLARE_bool(rerank);

// ============================================================================
// Passkey / benchmark parameters
// ============================================================================
DECLARE_int32(junk);
DECLARE_int32(pos);

// ============================================================================
// Imatrix parameters
// ============================================================================
DECLARE_string(output);
DECLARE_int32(output_frequency);
DECLARE_string(output_format);
DECLARE_int32(save_frequency);
DECLARE_bool(process_output);
DECLARE_bool(ppl);
DECLARE_int32(chunk);
DECLARE_bool(show_statistics);
DECLARE_bool(parse_special);

// ============================================================================
// Bench parameters
// ============================================================================
DECLARE_bool(pps);
DECLARE_bool(tgs);
DECLARE_string(npp);
DECLARE_string(ntg);
DECLARE_string(npl);

// ============================================================================
// Perplexity parameters
// ============================================================================
DECLARE_int32(ppl_stride);
DECLARE_int32(ppl_output_type);
DECLARE_bool(hellaswag);
DECLARE_int32(hellaswag_tasks);
DECLARE_bool(winogrande);
DECLARE_int32(winogrande_tasks);
DECLARE_bool(multiple_choice);
DECLARE_int32(multiple_choice_tasks);
DECLARE_bool(kl_divergence);
DECLARE_string(save_all_logits);

// ============================================================================
// Server parameters
// ============================================================================
DECLARE_string(host);
DECLARE_int32(port);
DECLARE_bool(reuse_port);
DECLARE_string(path);
DECLARE_string(api_prefix);
DECLARE_string(tools);
DECLARE_string(api_key);
DECLARE_string(api_key_file);
DECLARE_string(ssl_key_file);
DECLARE_string(ssl_cert_file);
DECLARE_string(chat_template_kwargs);
DECLARE_int32(timeout);
DECLARE_int32(sse_ping_interval);
DECLARE_int32(threads_http);
DECLARE_bool(cache_prompt);
DECLARE_int32(cache_reuse);
DECLARE_bool(metrics);
DECLARE_bool(props);
DECLARE_bool(slots);
DECLARE_string(slot_save_path);
DECLARE_bool(jinja);
DECLARE_string(reasoning_format);
DECLARE_string(reasoning);
DECLARE_int32(reasoning_budget);
DECLARE_string(reasoning_budget_message);
DECLARE_string(chat_template);
DECLARE_string(chat_template_file);
DECLARE_bool(skip_chat_parsing);
DECLARE_bool(prefill_assistant);
DECLARE_string(slot_prompt_similarity);
DECLARE_int32(sleep_idle_seconds);
DECLARE_bool(simple_io);

// ============================================================================
// Control vector generator parameters
// ============================================================================
DECLARE_string(positive_file);
DECLARE_string(negative_file);
DECLARE_int32(pca_batch);
DECLARE_int32(pca_iter);
DECLARE_string(method);

// ============================================================================
// Logging parameters (server-specific)
// ============================================================================
DECLARE_bool(log_disable);
DECLARE_bool(log_file_flag);
DECLARE_string(log_prompts_dir);
DECLARE_bool(log_colors);
DECLARE_bool(verbose);
DECLARE_bool(offline);
DECLARE_int32(verbosity);
DECLARE_bool(log_prefix);
DECLARE_bool(log_timestamps);

// ============================================================================
// Speculative decoding parameters
// ============================================================================
DECLARE_int32(spec_draft_threads);
DECLARE_int32(spec_draft_threads_batch);
DECLARE_string(spec_draft_cpu_mask);
DECLARE_string(spec_draft_cpu_range);
DECLARE_int32(spec_draft_cpu_strict);
DECLARE_int32(spec_draft_prio);
DECLARE_int32(spec_draft_poll);
DECLARE_string(spec_draft_cpu_mask_batch);
DECLARE_bool(spec_draft_cpu_range_batch);
DECLARE_int32(spec_draft_cpu_strict_batch);
DECLARE_int32(spec_draft_prio_batch);
DECLARE_int32(spec_draft_poll_batch);
DECLARE_string(spec_draft_type_k);
DECLARE_string(spec_draft_type_v);
DECLARE_string(spec_draft_override_tensor);
DECLARE_bool(spec_draft_cpu_moe);
DECLARE_int32(spec_draft_n_cpu_moe);
DECLARE_int32(spec_draft_n_max);
DECLARE_int32(spec_draft_n_min);
DECLARE_string(spec_draft_p_split);
DECLARE_string(spec_draft_p_min);
DECLARE_bool(spec_draft_backend_sampling);
DECLARE_string(spec_draft_device);
DECLARE_string(spec_draft_ngl);
DECLARE_string(spec_draft_model);
DECLARE_string(spec_type);
DECLARE_int32(spec_ngram_mod_n_min);
DECLARE_int32(spec_ngram_mod_n_max);
DECLARE_int32(spec_ngram_mod_n_match);
DECLARE_int32(spec_ngram_simple_size_n);
DECLARE_int32(spec_ngram_simple_size_m);
DECLARE_int32(spec_ngram_simple_min_hits);
DECLARE_int32(spec_ngram_map_k_size_n);
DECLARE_int32(spec_ngram_map_k_size_m);
DECLARE_int32(spec_ngram_map_k_min_hits);
DECLARE_int32(spec_ngram_map_k4v_size_n);
DECLARE_int32(spec_ngram_map_k4v_size_m);
DECLARE_int32(spec_ngram_map_k4v_min_hits);
DECLARE_bool(draft);
DECLARE_bool(draft_min);
DECLARE_bool(spec_ngram_size_n);
DECLARE_bool(spec_ngram_size_m);
DECLARE_bool(spec_ngram_min_hits);

// ============================================================================
// Lookup cache parameters
// ============================================================================
DECLARE_string(lookup_cache_static);
DECLARE_string(lookup_cache_dynamic);

// ============================================================================
// Vocoder parameters
// ============================================================================
DECLARE_string(model_vocoder);
DECLARE_bool(tts_use_guide_tokens);
DECLARE_string(tts_speaker_file);
DECLARE_bool(tts_oute_default);

// ============================================================================
// Diffusion parameters
// ============================================================================
DECLARE_int32(diffusion_steps);
DECLARE_bool(diffusion_visual);
DECLARE_string(diffusion_eps);
DECLARE_int32(diffusion_algorithm);
DECLARE_string(diffusion_alg_temp);
DECLARE_int32(diffusion_block_length);
DECLARE_string(diffusion_cfg_scale);
DECLARE_string(diffusion_add_gumbel_noise);

// ============================================================================
// Finetune parameters
// ============================================================================
DECLARE_string(learning_rate);
DECLARE_string(learning_rate_min);
DECLARE_string(learning_rate_decay_epochs);
DECLARE_string(weight_decay);
DECLARE_string(val_split);
DECLARE_int32(epochs);
DECLARE_string(optimizer);

// ============================================================================
// Debug parameters
// ============================================================================
DECLARE_bool(check);
DECLARE_bool(save_logits);
DECLARE_string(logits_output_dir);
DECLARE_string(tensor_filter);

// ============================================================================
// Preset defaults
// ============================================================================
DECLARE_bool(embd_gemma_default);
DECLARE_bool(spec_default);
