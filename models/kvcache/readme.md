# mooncake submodule
git submodule update --init --recursive --depth=1

# build
`mkdir build; cd build; cmake .. -DLHM_ENABLE_MOONCAKE=ON`

# run mini mooncake
* `mooncake_master` (simple default config)  
* how to build and run, more info in mooncake

# config use mooncake as kvcahe
`./llm_cli --model_path Qwen35.gguf --enable_mooncake true`
