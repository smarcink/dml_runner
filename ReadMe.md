# Test Case
|Engine| Type|Command line|
|----|---|---|
|DML|MHA|`cross_runner.exe --iters 1 --type mha_dml mha_opts --data_type fp16 --layout nchw --mha_type qkv --shape_input 2,64,8,3,160`|
|DML|QGEMM|`cross_runner.exe --type quant_gemm_dml quant_gemm_opts --layout nchw --data_type fp16 --quantize_data_type uint4 --shape_a 1,1,1,14336 --shape_b 1,1,4096,14336 --shape_c 1,1,1,4096 --b_transposed --b_quantized --block_size 32 --has_zeropoint`|

# Building

1. git clone
2. git submodule update --init --recursive
3. cmake

In case of CMake errors, especially those related to finding the OpenCL installation, you can try the following steps:

- Install the full GPU driver package from your GPU vendor's official website. This should include the OpenCL runtimes necessary for development.
- Download the OpenCL headers and loader from the Khronos Group's official GitHub repository or use the OpenCL SDK provided by your GPU vendor.
- If you encounter issues with CMake finding OpenCL, you can specify the paths to the OpenCL headers and library using the following command line (adjust the paths to match your OpenCL SDK installation):

```bash
cmake . -Bbuild -G "Visual Studio 17 2022" -DOpenCL_INCLUDE_DIR="C:\path\to\OpenCL\include" -DOpenCL_LIBRARY="C:\path\to\OpenCL\lib\opencl.lib"
```