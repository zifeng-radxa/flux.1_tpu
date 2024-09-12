#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

void bm_device_mem_offset(bm_device_mem_t* src, bm_device_mem_t* dst, int offset, int size){
    DEVICE_ASSERT(src->size >= offset + size);
    dst->u.device.device_addr = src->u.device.device_addr + offset;
    dst->size = size;
} 
void run_model(const char* model_path, const char* part_name, void** inputs, int input_num, void** outputs, int output_num, int device_id){
    // load model 
    bm_handle_t handle = get_handle(device_id);
    void* model = bmrt_create(handle);
    DEVICE_ASSERT(bmrt_load_bmodel(model, model_path)==true);
    const bm_net_info_t* net_info = bmrt_get_network_info(model, part_name);
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(output_num, sizeof(bm_tensor_t));
    bm_device_mem_t input_mems[input_num];
    bm_device_mem_t output_mems[output_num];
    loop(i, input_num){
        DEVICE_ASSERT( bm_malloc_device_byte(handle, input_mems+i, net_info->max_input_bytes[i]) == 0 );
        DEVICE_ASSERT( bm_memcpy_s2d(handle, input_mems[i], inputs[i]) == 0 );
        bmrt_tensor_with_device(input_tensors+i, input_mems[i], net_info->input_dtypes[i], net_info->stages[0].input_shapes[i]);
    }
    loop(i, output_num){
        DEVICE_ASSERT( bm_malloc_device_byte(handle, output_mems+i, net_info->max_output_bytes[i]) == 0 );
        bmrt_tensor_with_device(output_tensors+i, output_mems[i], net_info->output_dtypes[i], net_info->stages[0].output_shapes[i]);
    }
    FUNC_TIME_START;
    // run model
    auto ret = bmrt_launch_tensor_ex(model, part_name, input_tensors, input_num, output_tensors, output_num, true, false);
    DEVICE_ASSERT(ret);
    bm_thread_sync(handle);
    FUNC_TIME_END;
    // copy output data into cpu
    loop(i, output_num){
        DEVICE_ASSERT( bm_memcpy_d2s(handle, outputs[i], output_mems[i]) == 0 );
    }
    // free device memory
    loop(i, input_num){
        bm_free_device(handle, input_mems[i]);
    }
    loop(i, output_num){
        bm_free_device(handle, output_mems[i]);
    }
    free(input_tensors);
    free(output_tensors);
    // free model 
    bmrt_destroy(model);
    bm_dev_free(handle);
}

bm_handle_t get_handle(int device_id){
    bm_handle_t bm_handle;
    bm_dev_request(&bm_handle, device_id);
    return bm_handle;
}

void copy_tensor_into_cpu(void* tensor, void* data, int device){
    // tensor is a pointer to bm_device_mem_t (user need to check the size)
    // data is a pointer to cpu memory
    bm_device_mem_t* bm_tensor = (bm_device_mem_t*)tensor;
    bm_handle_t handle = get_handle(device);
    bm_memcpy_d2s(handle, data, *bm_tensor);
}

#ifdef __cplusplus
}
#endif
