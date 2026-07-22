# lhm
Local high model.    
Focus on user local device to run one model.     

# Inspired by llama.cpp
1. Fork from llama.cpp
2. Trim some old model
3. Use third party library
4. Just support one model to run
5. Merge some code from llama.cpp by time

# build
#### linux cpu
```linux
cmake -B build -DGGML_CUDA=OFF
cmake --build build --config Release -j 8
```

#### linux cuda
```linux cuda
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 8
```

#### mac
```
cmake -B build -DGGML_CUDA=OFF -DLLAMA_METAL=ON
cmake --build build --config Release -j 8
```

#### windows

#### build with mooncake
[mooncake as kvcache](models/kvcache/readme.md)

# run
```
./bin/llm_cli --model /modelpath/model.gguf --log_level info
```
