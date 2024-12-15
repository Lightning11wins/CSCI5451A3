/* Borrowed from http://computer-graphics.se/hello-world-for-cuda.html
 * This program takes the string "Hello ", prints it, then passes it to CUDA
 * with an array * of offsets. Then the offsets are added in parallel to
 * produce the string "World!"
 * By Ingemar Ragnemalm 2010
 */

#include <stdlib.h>
#include <stdio.h>

int const N = 16;
int const blocksize = 16;

__global__
void hello(char* const a, int const* const b) {
  a[threadIdx.x] += b[threadIdx.x];
}


int main(int argc, char ** argv) {
  char a[N] = "Hello \0\0\0\0\0\0";
  int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  char* gpu_a;
  int* gpu_b;
  int const a_size = N * sizeof(char);
  int const b_size = N * sizeof(int);

  printf("%s", a);
 
  cudaMalloc((void**) &gpu_a, a_size);
  cudaMalloc((void**) &gpu_b, b_size);

  cudaMemcpy(gpu_a, a, a_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_b, b, b_size, cudaMemcpyHostToDevice);

  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);
  hello<<<dimGrid, dimBlock>>>(gpu_a, gpu_b);

  cudaMemcpy(a, gpu_a, csize, cudaMemcpyDeviceToHost);
  cudaFree(gpu_a);

  printf("%s\n", a);

  return 0;
}