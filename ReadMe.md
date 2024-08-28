# Test Case
|Engine| Type|Command line|
|----|---|---|
|DML|MHA|`cross_runner.exe --iters 1 --type mha_dml mha_opts --data_type fp16 --layout nchw --mha_type qkv --shape_input 2,64,8,3,160`|
|DML|QGEMM|`cross_runner.exe --type quant_gemm_dml quant_gemm_opts --layout nchw --data_type fp16 --quantize_data_type uint4 --shape_a 1,1,1,14336 --shape_b 1,1,4096,14336 --shape_c 1,1,1,4096 --b_transposed --b_quantized --block_size 32 --has_zeropoint`|
