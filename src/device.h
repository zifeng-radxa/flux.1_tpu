
#pragma once

#include "bmruntime.h"
#include "bmruntime_interface.h"
#include "bmodel.hpp"
#include "help.h"

#define DEVICE_ASSERT(x) \
    do { \
        if (!(x)) { \
            printf("Error: %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif



bm_handle_t get_handle(int device_id);
void copy_tensor_into_cpu(void* tensor, void* data, int device);
void run_model(const char* model_path, const char* part_name, void** inputs, int input_num, void** outputs, int output_num, int device_id);
void bm_device_mem_offset(bm_device_mem_t* src, bm_device_mem_t* dst, int offset, int size);



#ifdef __cplusplus
}
#endif


