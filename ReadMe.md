# Test Case
|Engine| Type|Command line|
|----|---|---|
|DML|MHA|`cross_runner.exe --type mha_dml mha_opts --mha_type qkv --data_type fp16 --layout ncdhw --shape_input_qkv 2,4096,8,3,40`|
[DML|MHA\'cross_runner.exe --type mha_dml mha_opts --mha_type q_kv --data_type fp16 --layout ncdhw --shape_input_q 2,4096,320 --shape_input_kv 2,4096,8,2,40'|
|DML|QGEMM|`cross_runner.exe --type quant_gemm_dml quant_gemm_opts --layout nchw --data_type fp16 --quantize_data_type uint4 --shape_a 1,1,1,14336 --shape_b 1,1,4096,14336 --shape_c 1,1,1,4096 --b_transposed --b_quantized --block_size 32 --has_zeropoint`|

## *stateless kernel support
1. add --use_stateless in your command line, like:
   
```bash
# GEMM test case
./cross_runner.exe --iters 1  --type gemm_cm gemm_opts --gemm_type ab  --data_type fp16 --layout nchw --shape_a 1,1,1024,16 --shape_b 1,1,16,1024 --b_managed gemm_cm_opts --large_grf --tile_m 1 --tile_k 16 --tile_n 16 --slice_k 1 --dump_asm --use_stateless 

```
2. copy stateless version kernel to crossrunner binary folder, now we have two samples:
```bash
tools\cross_runner\kernels\gemm_nchw_fp16_stateless.cpp
```

# Building

1. git clone
2. git submodule update --init --recursive
3. cmake

In case of CMake errors, especially those related to finding the OpenCL installation, you can try the following steps:

- Install the full GPU driver package from your GPU vendor's official website. This should include the OpenCL runtimes necessary for development.
- Download the OpenCL headers and loader from the Khronos Group's official GitHub repository or use the OpenCL SDK provided by your GPU vendor.
- If you encounter issues with CMake finding OpenCL, you can specify the paths to the OpenCL headers and library using the following command line (adjust the paths to match your OpenCL SDK installation):

```bash
cmake . -Bbuild -G "Visual Studio 17 2022" -DOpenCL_INCLUDE_DIR="C:\Users\yarudu\Documents\project\third-party\OpenCL-SDK-v2024.05.08-Win-x64\include" -DOpenCL_LIBRARY="C:\Users\yarudu\Documents\project\third-party\OpenCL-SDK-v2024.05.08-Win-x64\lib\opencl.lib"

cd build
# the default build was Debug version
cmake  --build .  -j

#if your Debug version notwork, pls try Build Release version
cmake  --build . --config Release -j
```