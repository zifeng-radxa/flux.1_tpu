#include "ops.h"

#ifdef __cplusplus
extern "C" {
#endif


void ops_rmsnorm_inplace(float* input, float* weight, int batch_size, int feature_size, float eps) {
    FUNC_TIME_START;
    loop(i, batch_size) {
        float sum_of_squares = 0.0;
        loop(j, feature_size) {
            size_t idx = i * feature_size + j;
            sum_of_squares += input[idx] * input[idx];
        }
        float inv_rms = fast_inverse_sqrt(sum_of_squares / feature_size + eps);

        loop(j, feature_size) {
            input[i * feature_size + j] *= weight[j] * inv_rms;
        }
    }
    FUNC_TIME_END;
}

void reorder_inverse(void* input, void* output, int n, int h, int w, int p, int q, int c, int dtype_len) {
    int cur_idx = 0;
    int t;
    int target_idx = 0;
    int unloop = 32 / dtype_len;
    // 0.4ms in x86 161 server
    loop(ni, n) {
        loop(ci, c){
            loop(hi, h){
                loop(pi, p){
                    loop(t, w*q){
                        target_idx = (((((ni*c+ci)*h+hi)*p+pi)*w*q+t));
                        cur_idx = ((((((ni*h+hi)*w+t/q)*p+pi)*q+t%q)*c+ci));
                        memcpy((char*)output + target_idx * dtype_len,
                               (char*)input + cur_idx * dtype_len, dtype_len);
                    }
                }
            }
        }
    }
}

inline void reorder_basic(void* input, void* output, int n, int h, int w, int p, int q, int c, int dtype_len) {
    // nhwpqc -> nchpwq
    int cur_idx = 0;
    int target_idx = 0;
    // Fallback to non-AVX implementation
    // need compiler with O3
    // experiments show that only macos can acclerate by SIMD (without test with shuffle)
    // for 1*640*640*2*2*16, the speedup is 1ms(airbox) and 1ms(x86 161 server)
    loop(ni, n) {
        loop(hi, h) {
            loop(wi, w) {
                loop(pi, p) {
                    loop(qi, q) {
                        loop(ci, c) {
                            cur_idx    = ((((((ni*h+hi)*w+wi)*p+pi)*q+qi)*c+ci));
                            target_idx = ((((((ni*c+ci)*h+hi)*p+pi)*w+wi)*q+qi));
                            memcpy((char*)output + target_idx * dtype_len, (char*)input + cur_idx * dtype_len, dtype_len);
                        }
                    }
                }
            }
        }
    }
}

void convert_pathify_to_latent(void* input, void* output, int batch, int width, int height, int patch1, int patch2, int channel, int dtype_len){
    // hwppc -> cwphp
    FUNC_TIME_START;
#if defined(__aarch64__)
// if macos 
    #if defined(__APPLE__)
        printf("more to come\n");
    #else
        reorder_basic(input, output, batch, height, width, patch1, patch2, channel, dtype_len);
    #endif
#elif defined(__x86_64__)
    reorder_inverse(input, output, batch, height, width, patch1, patch2, channel, dtype_len);
#else 
    printf("Unsupported platform\n");
#endif
    FUNC_TIME_END;
}

void ops_safe_rmsnorm(float* input, float* output, float* weight, int batch_size, int feature_size, float eps) {
    for (int i = 0; i < batch_size; i++) {
        float sum_of_squares = 0.0;
        for (int j = 0; j < feature_size; j++) {
            float product = input[i * feature_size + j] * input[i * feature_size + j];
            if (product > FLT_MAX - sum_of_squares) {
                product = FLT_MAX - sum_of_squares;
            }
            sum_of_squares += product;
        }
        float inv_rms = fast_inverse_sqrt(sum_of_squares / feature_size + eps);

        for (int j = 0; j < feature_size; j++) {
            float product = weight[j] * (input[i * feature_size + j] * inv_rms);
            if (product > FLT_MAX) {
                product = FLT_MAX;
            }
            output[i * feature_size + j] = product;
        }
    }
}

void ops_rmsnorm(float* input, float* output, float* weight, int batch_size, int feature_size, float eps) {
    FUNC_TIME_START;
    loop(i, batch_size) {
        float sum_of_squares = 0.0;
        loop(j, feature_size) {
            sum_of_squares += input[i * feature_size + j] * input[i * feature_size + j];
        }
        float inv_rms = fast_inverse_sqrt(sum_of_squares / feature_size + eps);

        loop(j, feature_size) {
            output[i * feature_size + j] = weight[j] * (input[i * feature_size + j] * inv_rms);
        }
    }
    FUNC_TIME_END;
}



#ifdef __cplusplus
}
#endif
