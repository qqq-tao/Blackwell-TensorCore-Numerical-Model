import re

# from tkinter import N

dense_mma_pattern = """
__global__ void mma_sync_aligned_m{M}n{N}k{K}_row_col_{output_type}_{A_type}_{B_type}_{output_type}(int *A, int *B, int *C, int *D) {{
    int tidx = threadIdx.x;
    
    int A_offset = {A_offset_string};   
    int B_offset = {B_offset_string};   
    int C_offset = {C_offset_string};    
    int D_offset = C_offset;
    
    asm volatile(
        "mma.sync.aligned.m{M}n{N}k{K}.row.col.{output_type}.{A_type}.{B_type}.{output_type}{boolen_modifier} "
        "{D_registers}, "
        "{A_registers}, "
        "{B_registers}, "
        "{C_registers};"
        : {D_registers_input}
        : {A_registers_input}
          {B_registers_input}
          {C_registers_input}
    );
    
}}
"""

dense_block_scale_mma_pattern = """
#include <stdint.h>
__global__ void mma_sync_aligned_m{M}n{N}k{K}_row_col_{output_type}_{A_type}_{B_type}_{output_type}_{stype}(int *A, int *B, int *C, int *D, int *scale_A, int *scale_B) {{
    int tidx = threadIdx.x;
    
    int A_offset = {A_offset_string};   
    int B_offset = {B_offset_string};   
    int C_offset = {C_offset_string};    
    int D_offset = C_offset;
        
    uint32_t scaleAData = scale_A[tidx / 4 + (tidx % 4) % 2 * 8];
    uint32_t scaleBData = scale_B[tidx / 4 + (tidx % 4) % 2 * 8];
    
    uint16_t tidA = {tidA};
    uint16_t bidA = {bidA};
    uint16_t tidB = {tidB};
    uint16_t bidB = {bidB};
    
    asm volatile(
        "mma.sync.aligned.m{M}n{N}k{K}.row.col.{kind}.block_scale.{scale_vec_size}.{output_type}.{A_type}.{B_type}.{output_type}.{stype} "
        "{D_registers}, "
        "{A_registers}, "
        "{B_registers}, "
        "{C_registers}, "
        "{A_scale_register}, "
        "{A_index_registers}, "
        "{B_scale_register}, "
        "{B_index_registers};"
        : {D_registers_input}
        : {A_registers_input}
          {B_registers_input}
          {C_registers_input}
          "r"(uint32_t(scaleAData)), "h"(bidA), "h"(tidA),
          "r"(uint32_t(scaleBData)), "h"(bidB), "h"(tidB)
    );
    
}}
"""
# int N = 8;
# int K = 8;

# if (tidx % 32 == 0){{
#     printf("A: ");
#     for(int i = 0 ; i< K;i++){{
#         printf("%08x ", A[i]);
#     }}
#     printf("\\n");
#     printf("B:");

#     for(int i = 0 ; i< K;i++){{
#         printf("%08x ", B[i]);
#     }}
# }}

# if (tidx % 32 ==0){{

# printf("%08x ", C[0]);
# printf("%08x ", D[0]);
# }}
bit_length = {
    "f64": 64,  # 64-bit floating point
    "tf32": 32,  # TensorFloat32, typically 32-bit
    "f32": 32,  # Float32, typically 32-bit
    "s32": 32,  # Int32, typically 32-bit
    "bf16": 16,  # bfloat16, 16-bit
    "f16": 16,  # 16-bit floating point
    # Exponent 4 bits, Mantissa 3 bits (posits-like, 7 bits total)
    "e4m3": 8,
    # Exponent 5 bits, Mantissa 2 bits (posits-like, 8 bits total)
    "e5m2": 8,
    "e2m1": 4,
    "u8": 8,  # Unsigned 8-bit integer
    "s8": 8,  # Signed 8-bit integer
    "u4": 4,  # Unsigned 4-bit integer
    "s4": 4,  # Signed 4-bit integer
    "b1": 1,  # 1-bit boolean
}

cuda_func_pattern = "mma_sync_aligned_m{M}n{N}k{K}_row_col_{output_type}_{A_type}_{B_type}_{output_type}"
cuda_block_scale_func_pattern = "mma_sync_aligned_m{M}n{N}k{K}_row_col_{output_type}_{A_type}_{B_type}_{output_type}_{stype}"


def initialize_registers():
    global prev, register_accumulator
    prev = 0
    register_accumulator = 0


def generate_registers(num_registers):
    global prev, register_accumulator
    prev = 0
    register_str = "{"
    for i in range(int(num_registers)):
        if prev == 1:
            register_str += ", "

        register_str += f"%{register_accumulator}"
        register_accumulator += 1
        prev = 1  # Mark that we have used at least one register

    register_str += "}"
    return register_str


def generate_registers_input(
    M,
    N,
    K,
    _M,
    _N,
    _K,
    A_tile_shape,
    B_tile_shape,
    C_tile_shape,
    bit_len_A_type,
    bit_len_C_type,
    params,
):
    register_input_str = ""

    prev = 0

    A_GPR_num_per_Tile = params["A_GPR_num_per_Tile"]
    B_GPR_num_per_Tile = params["B_GPR_num_per_Tile"]
    C_GPR_num_per_Tile = params["C_GPR_num_per_Tile"]

    ldK_len = int(K * bit_len_A_type / 32)
    ldN_len = int(N * bit_len_C_type / 32)

    # coordination of every thread, (tidx/ thread_num_per_row, tidx % thread_num_per_row)
    A_thread_num_per_row = int(_K * bit_len_A_type / (32 * A_GPR_num_per_Tile))
    A_offset_string = f"(tidx / {A_thread_num_per_row})* {int(ldK_len)} + (tidx % {A_thread_num_per_row}) * ({A_GPR_num_per_Tile})"

    B_thread_num_per_col = int(_K * bit_len_A_type / (32 * B_GPR_num_per_Tile))
    B_offset_string = f"(tidx / {B_thread_num_per_col})* {ldK_len} + (tidx % {B_thread_num_per_col}) * ({B_GPR_num_per_Tile})"

    C_thread_num_per_row = int(N * bit_len_C_type / (32 * C_GPR_num_per_Tile))
    C_offset_string = f"(tidx / {C_thread_num_per_row})* {ldN_len} + (tidx % {C_thread_num_per_row}) * ({C_GPR_num_per_Tile})"

    params["A_offset_string"] = A_offset_string
    params["B_offset_string"] = B_offset_string
    params["C_offset_string"] = C_offset_string

    for A_tile_y in range(A_tile_shape[1]):
        for A_tile_x in range(A_tile_shape[0]):
            tile_base_offset = (
                A_tile_x * (K * _M) * bit_len_A_type / 32
            ) + A_tile_y * (_K * bit_len_A_type / 32)
            for GPR_rank in range(int(A_GPR_num_per_Tile)):
                register_input_str += (
                    f'"r"(A[A_offset + {int(GPR_rank+tile_base_offset)}])'
                )
                register_input_str += ", "
    params["A_registers_input"] = register_input_str

    c_register_inputs = []
    for C_tile_x in range(C_tile_shape[0]):
        for C_tile_y in range(C_tile_shape[1]):
            tile_base_offset = (
                C_tile_x * (N * _M) * bit_len_C_type / 32
            ) + C_tile_y * (_N * bit_len_C_type / 32)
            for GPR_rank in range(int(C_GPR_num_per_Tile)):
                c_register_inputs.append(
                    f'"r"(C[C_offset + {int(GPR_rank+tile_base_offset)}])'
                )
    params["C_registers_input"] = ", ".join(c_register_inputs)
    register_input_str = ""

    for B_tile_y in range(B_tile_shape[0]):
        for B_tile_x in range(B_tile_shape[1]):
            tile_base_offset = (
                B_tile_x * (K * _N) * bit_len_A_type / 32
            ) + B_tile_y * (_K * bit_len_A_type / 32)
            for GPR_rank in range(int(A_GPR_num_per_Tile)):
                register_input_str += (
                    f'"r"(B[B_offset + {int(GPR_rank+tile_base_offset)}])'
                )
                register_input_str += ", "
    params["B_registers_input"] = register_input_str

    register_input_str = ""
    prev = 0
    for C_tile_x in range(C_tile_shape[0]):
        for C_tile_y in range(C_tile_shape[1]):
            tile_base_offset = (
                C_tile_x * (N * _M) * bit_len_C_type / 32
            ) + C_tile_y * (_N * bit_len_C_type / 32)
            for GPR_rank in range(int(C_GPR_num_per_Tile)):
                if prev:
                    register_input_str += ", "
                register_input_str += (
                    f'"=r"(D[D_offset + {int(GPR_rank+tile_base_offset)}])'
                )
                prev = 1
    params["D_registers_input"] = register_input_str
    prev = 0


def fill_in_params(
    M,
    N,
    K,
    minimum_mma,
    A_type,
    B_type,
    C_type,
    bit_len_A_type,
    bit_len_C_type,
    is_block_scale,
    scale_index: dict,
    stype,
):

    global params
    params = {}
    params["A_type"] = A_type
    params["B_type"] = B_type
    params["output_type"] = C_type

    if is_block_scale:

        params["stype"] = stype
        if K == 32:
            params["scale_vec_size"] = "scale_vec::1X"
            params["kind"] = "kind::mxf8f6f4"
        elif stype == "ue8m0":
            params["scale_vec_size"] = "scale_vec::2X"
            params["kind"] = "kind::mxf4nvf4"
        else:
            params["scale_vec_size"] = "scale_vec::4X"
            params["kind"] = "kind::mxf4nvf4"

        params["tidA"] = scale_index["tidA"]
        params["bidA"] = scale_index["bidA"]
        params["tidB"] = scale_index["tidB"]
        params["bidB"] = scale_index["bidB"]
    else:
        if A_type == "b1":
            params["boolen_modifier"] = ".and.popc"
        else:
            params["boolen_modifier"] = ""

    params["M"] = M
    params["N"] = N
    params["K"] = K

    A_GPR_num = bit_len_A_type * M * K / (32 * 32)
    B_GPR_num = bit_len_A_type * N * K / (32 * 32)
    C_GPR_num = bit_len_C_type * M * N / (32 * 32)
    # print(f"A_GPR_num: {A_GPR_num}, B_GPR_num: {B_GPR_num}, C_GPR_num: {C_GPR_num}")

    # For each matrix type, generate its corresponding registers based on GPR count
    initialize_registers()
    params["D_registers"] = generate_registers(C_GPR_num)
    params["A_registers"] = generate_registers(A_GPR_num)
    params["B_registers"] = generate_registers(B_GPR_num)
    params["C_registers"] = generate_registers(C_GPR_num)
    if is_block_scale:
        params["A_scale_register"] = generate_registers(1)
        params["A_index_registers"] = generate_registers(2)
        params["B_scale_register"] = generate_registers(1)
        params["B_index_registers"] = generate_registers(2)

    params["_M"], params["_N"], params["_K"] = (
        minimum_mma[0],
        minimum_mma[1],
        minimum_mma[2],
    )
    _M, _N, _K = minimum_mma[0], minimum_mma[1], minimum_mma[2]

    # coordination = (M/_M, N/_N, K/_K)  # minimum块所在坐标
    A_tile_shape = (int(M / _M), int(K / _K))
    B_tile_shape = (int(K / _K), int(N / _N))
    C_tile_shape = (int(M / _M), int(N / _N))

    params["A_GPR_num_per_Tile"] = A_GPR_num / (A_tile_shape[0] * A_tile_shape[1])
    params["B_GPR_num_per_Tile"] = B_GPR_num / (B_tile_shape[0] * B_tile_shape[1])
    params["C_GPR_num_per_Tile"] = C_GPR_num / (C_tile_shape[0] * C_tile_shape[1])

    generate_registers_input(
        M,
        N,
        K,
        _M,
        _N,
        _K,
        A_tile_shape,
        B_tile_shape,
        C_tile_shape,
        bit_len_A_type,
        bit_len_C_type,
        params,
    )

    template = dense_block_scale_mma_pattern if is_block_scale else dense_mma_pattern
    placeholders = re.findall(r"{(\w+)}", template)
    missing_params = [p for p in placeholders if p not in params]
    if missing_params:
        print(f"Missing parameters: {missing_params}")
    if is_block_scale:
        generated_code = dense_block_scale_mma_pattern.format(**params)
    else:
        generated_code = dense_mma_pattern.format(**params)
    return generated_code


if __name__ == "__main__":
    M = 16
    N = 8
    K = 32
    A_type = "e4m3"
    B_type = "e2m3"
    C_type = "f32"
    bit_len_A_type = 8
    bit_len_C_type = 32
    is_block_scale = True
    stype = "ue8m0"  # or "e4m3", "e5m2", etc.
    minimum_mma = (8, 8, 32)
    scale_index = {
        "tidA": 0,
        "bidA": 0,
        "tidB": 0,
        "bidB": 0,
    }

    cuda_code = fill_in_params(
        M=M,
        N=N,
        K=K,
        minimum_mma=minimum_mma,
        A_type=A_type,
        B_type=B_type,
        C_type=C_type,
        bit_len_A_type=bit_len_A_type,
        bit_len_C_type=bit_len_C_type,
        is_block_scale=is_block_scale,
        scale_index=scale_index,
        stype=stype,
    )
    print(cuda_code)

    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    mod = SourceModule(cuda_code, arch="sm_120a")
    mma_sync_func = mod.get_function(
        cuda_block_scale_func_pattern.format(
            M=M,
            N=N,
            K=K,
            output_type=C_type,
            A_type=A_type,
            B_type=B_type,
            stype=stype,
        )
    )
