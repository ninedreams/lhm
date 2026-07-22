## mooncake submodule
git submodule update --init --recursive --depth=1

## build
```
cmake -B build -DGGML_CUDA=ON -DLHM_ENABLE_MOONCAKE=ON
cmake --build build --config Release -j 8
```

## run mini mooncake
```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8081 \
  --default_kv_lease_ttl=5000
```
* how to build and run, more info in [mooncake](https://github.com/kvcache-ai/Mooncake)

## config use mooncake as kvcahe
`./llm_cli --model_path Qwen35.gguf --enable_mooncake true`
