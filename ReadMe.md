# Test Case
|Engine| Type|Command line|
|----|---|---|
|DML|MHA|`cross_runner.exe --iters 1 --type mha_dml mha_opts --data_type fp16 --layout nchw --mha_type qkv --shape_input 2,64,8,3,160`|
|CM|GEMM_QK_QKV|`cross_runner.exe --iters 100 --no_conform=0 --type gemm_cm gemm_opts --gemm_type qk_qkv --data_type fp16 --layout nchw --shape_a 2,64,8,3,160 gemm_cm_opts --large_grf --tile_m 16 --tile_k 80 --tile_n 64 --lws_x 1 --lws_y 1 --lws_z 2 --slice_k 2 --dump_asm`|
|CM|GEMM_QK_QKV_DPAS|`cross_runner.exe --iters 100 --no_conform=0 --type gemm_cm gemm_opts --gemm_type qk_qkv_dpas --data_type fp16 --layout nchw --shape_a 2,64,8,3,160 gemm_cm_opts --large_grf  --tile_m 16 --tile_k 80 --tile_n 64 --lws_x 8 --lws_y 2 --lws_z 1 --dump_asm`|
|CM|GEMM_AB| `cross_runner.exe --iters 100 --no_conform=0 --type gemm_cm gemm_opts --gemm_type ab --data_type fp16 --layout nchw --shape_a 1,1,4096,320 --shape_b 1,1,320,320 --b_managed gemm_cm_opts --large_grf --tile_m 16 --tile_k 32 --tile_n 32 --dump_asm`|
|DML|GEMM_AB| `cross_runner.exe --iters 100 --type gemm_dml gemm_opts --data_type fp16 --layout nchw --gemm_type ab --shape_a 1,1,4096,320  --shape_b 1,1,320,320 --b_managed`|