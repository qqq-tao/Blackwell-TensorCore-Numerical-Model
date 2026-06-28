#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>  // For uint32_t
#include <stdio.h>   // 包含 printf 的头文件

#include <fstream>
#include <iostream>
#include <vector>
#define CUDA_CHECK(FN)                                                        \
  {                                                                           \
    auto cudaError = FN;                                                      \
    if (cudaError != cudaSuccess) {                                           \
      std::cerr << "FATAL: " #FN " failed: " << cudaGetErrorString(cudaError) \
                << std::endl;                                                 \
      return -1;                                                              \
    }                                                                         \
  }

__global__ void mma_sync_aligned_m16n8k8_row_col_f32_tf32_tf32_f32(int *A,
                                                                   int *B,
                                                                   int *C,
                                                                   int *D) {
  int tidx = threadIdx.x;

  int A_offset = (tidx / 4) * 8 + (tidx % 4) * (1.0);
  int B_offset = (tidx / 4) * 8 + (tidx % 4) * (1.0);
  int C_offset = (tidx / 4) * 8 + (tidx % 4) * (2.0);
  int D_offset = C_offset;

  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};"
      : "=r"(D[D_offset + 0]), "=r"(D[D_offset + 1]), "=r"(D[D_offset + 64]),
        "=r"(D[D_offset + 65])
      : "r"(A[A_offset + 0]), "r"(A[A_offset + 64]), "r"(A[A_offset + 4]),
        "r"(A[A_offset + 68]), "r"(B[B_offset + 0]), "r"(B[B_offset + 4]),
        "r"(C[C_offset + 0]), "r"(C[C_offset + 1]) "r"(C[C_offset + 64]),
        "r"(C[C_offset + 65]));
}

int main() {
  int M = 16;
  int N = 8;
  int K = 8;
  uint32_t *h_A, *h_B, *h_C, *h_D;
  uint32_t *d_A, *d_B, *d_C, *d_D;

  h_A = (uint32_t *)malloc(sizeof(uint32_t) * M * K);
  h_B = (uint32_t *)malloc(sizeof(uint32_t) * N * K);
  h_C = (uint32_t *)malloc(sizeof(uint32_t) * M * N);
  h_D = (uint32_t *)malloc(sizeof(uint32_t) * M * N);

  cudaMalloc(&d_A, sizeof(uint32_t) * M * K);
  cudaMalloc(&d_B, sizeof(uint32_t) * N * K);
  cudaMalloc(&d_C, sizeof(uint32_t) * M * N);
  cudaMalloc(&d_D, sizeof(uint32_t) * M * N);
  cudaMalloc(&d_A, sizeof(uint32_t) * M * K);
  cudaMalloc(&d_B, sizeof(uint32_t) * N * K);
  cudaMalloc(&d_C, sizeof(uint32_t) * M * N);
  cudaMalloc(&d_D, sizeof(uint32_t) * M * N);
  memset(h_A, 0, sizeof(uint32_t) * M * K);
  memset(h_B, 0, sizeof(uint32_t) * N * K);
  memset(h_C, 0, sizeof(uint32_t) * M * N);

  h_A[0] = 0x3FFFE000;
  h_A[1] = 0x3FFFC000;
  h_A[2] = 0x3FFFE000;
  h_A[3] = 0x3FFFE000;
  h_A[4] = 0x00000000;
  h_A[5] = 0x00000000;
  h_A[6] = 0x00000000;
  h_A[7] = 0x00000000;

  h_B[0] = 0x3F800000;
  h_B[1] = 0x3A000000;
  h_B[2] = 0x35000000;
  h_B[3] = 0xBF800000;  // -1
  h_B[4] = 0x00000000;
  h_B[5] = 0x00000000;
  h_B[6] = 0x00000000;
  h_B[7] = 0x00000000;

  h_C[0] = 0x00000000;

  cudaMemcpy(d_A, h_A, sizeof(uint32_t) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(uint32_t) * N * K, cudaMemcpyHostToDevice);
  CUDA_CHECK(
      cudaMemcpy(d_C, h_C, sizeof(uint32_t) * M * N, cudaMemcpyHostToDevice));

  dim3 gridDim(1, 1, 1);
  dim3 blockDim(32, 1, 1);

  mma_sync_aligned_m16n8k8_row_col_f32_tf32_tf32_f32<<<gridDim, blockDim>>>(
      (int *)d_A, (int *)d_B, (int *)d_C, (int *)d_D);

  CUDA_CHECK(
      cudaMemcpy(h_D, d_D, sizeof(uint32_t) * M * N, cudaMemcpyDeviceToHost));

  printf("h_D[0]: %08x\n", h_D[0]);
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
