#ifndef TESTS_H
#define TESTS_H

#include "utility.hpp"
#include "workspace.hpp"
#include "initializer.hpp"
#include "equations.hpp"
#include "observer.hpp"

#ifndef DISABLE_CUDA
#include "cuda_wrapper.cuh"
#include "equations_cuda.cuh"
#endif

void test_cuda_00(void);
void test_cuda_01(void);
void test_cuda_02(void);
void test_cuda_03(void);
void test_cuda_04(void);
void test_cuda_05(void);
void test_cuda_06(void);
void test_cuda_07(void);
void test_cuda_08(void);
void test_cuda_09(void);
void test_cuda_10(void);
void test_cuda_11(void);
void test_cuda_12(void);
void test_cuda_13(void);
void test_cuda_14(void);

void test_cuda_15(void);
void test_cuda_16(void);
void test_cuda_17(void);
void test_cuda_18(void);
void test_cuda_19(void);

void test_00(void);
void test_01(void);
void test_02(void);
void test_03(void);
void test_04(void);

#endif
