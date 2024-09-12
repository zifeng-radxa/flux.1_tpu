#pragma once
#include <float.h>  // for FLT_MAX
#include "help.h"

#ifdef __cplusplus
extern "C" {
#endif
// neural network operations
void ops_rmsnorm(float* input, float* output, float* weight, int batch_size, int feature_size, float eps);
void ops_safe_rmsnorm(float* input, float* output, float* weight, int batch_size, int feature_size, float eps);
void ops_rmsnorm_inplace(float* input, float* weight, int batch_size, int feature_size, float eps) ;

// reorder
void reorder_inverse(void* input, void* output, int n, int h, int w, int p, int q, int c, int dtype_len) ;
inline void reorder_basic(void* input, void* output, int n, int h, int w, int p, int q, int c, int dtype_len);
void convert_pathify_to_latent(void* input, void* output, int batch, int width, int height, int patch1, int patch2, int channel, int dtype_len);

#ifdef __cplusplus
}
#endif
