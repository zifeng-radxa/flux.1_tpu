#include "model.h"

#define T5_PREFIX       "t5_"
#define T5_NUM_BLOCKS     24
#define T5_LAST_RMS_NORM_WEIGHT_SIZE 4096

#ifdef __cplusplus
extern "C" {
#endif

struct t5_encoder {
    int device_id;
    void* t5_model;
    bm_device_mem_t input_tokens;
    bm_device_mem_t hidden;
    bm_handle_t handle;
    const bm_net_info_t* net_head;
    const bm_net_info_t* net_tail;
    const bm_net_info_t* net_block0;
    float* weight_buffer;
    bool use_cpu_weight;
};

struct t5_encoder * t5_encoder_init(const char* filename, int device_id, const char* weight_file);
inline int t5_encoder_head(struct t5_encoder *encoder);
inline int t5_encoder_tail(struct t5_encoder *encoder);
inline int t5_encoder_block(struct t5_encoder *encoder, int block_idx);
int t5_encoder_free(struct t5_encoder *encoder);
int t5_encoder_run(struct t5_encoder *encoder, void* data, void* output);


struct t5_encoder * t5_encoder_init(const char* filename, int device_id, const char* weight_file){
    struct t5_encoder *encoder = (struct t5_encoder *)calloc(1, sizeof(struct t5_encoder));
    encoder->handle     = get_handle(device_id);
    encoder->device_id  = device_id;
    encoder->t5_model   = bmrt_create(encoder->handle);
    SD3_ASSERT(bmrt_load_bmodel(encoder->t5_model, filename)==true);
    encoder->net_head   = bmrt_get_network_info(encoder->t5_model, T5_PREFIX"head");
    encoder->net_tail   = bmrt_get_network_info(encoder->t5_model, T5_PREFIX"tail");
    encoder->net_block0 = bmrt_get_network_info(encoder->t5_model, T5_PREFIX"block_0");
    SD3_ASSERT( bm_malloc_device_byte(encoder->handle, &encoder->input_tokens, encoder->net_head->max_input_bytes[0]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(encoder->handle, &encoder->hidden, encoder->net_head->max_output_bytes[0]) == 0 );
    encoder->use_cpu_weight = true;
    size_t weight_size = T5_LAST_RMS_NORM_WEIGHT_SIZE * sizeof(float);
    encoder->weight_buffer = (float*) calloc(1, weight_size);
    // read_buffer_from_file(weight_file, weight_size, 0, encoder->weight_buffer);
    // free(net_names);
    return encoder;
}

inline int t5_encoder_head(struct t5_encoder *encoder){
    FUNC_TIME_START;
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(encoder->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(encoder->net_head->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors, encoder->input_tokens, encoder->net_head->input_dtypes[0], encoder->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(output_tensors, encoder->hidden, encoder->net_head->output_dtypes[0], encoder->net_head->stages[0].output_shapes[0]);
    auto ret = bmrt_launch_tensor_ex(encoder->t5_model, encoder->net_head->name, input_tensors, encoder->net_head->input_num, output_tensors, encoder->net_head->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(encoder->handle);
    free(input_tensors);
    free(output_tensors);
    FUNC_TIME_END;
    return 0;
}

inline int t5_encoder_block(struct t5_encoder *encoder, int block_idx){
    FUNC_TIME_START;
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(encoder->net_block0->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(encoder->net_block0->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,    encoder->hidden,  encoder->net_block0->input_dtypes[0],  encoder->net_block0->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(output_tensors,   encoder->hidden,  encoder->net_block0->output_dtypes[0], encoder->net_block0->stages[0].output_shapes[0]);
    char name[100];
    sprintf(name, T5_PREFIX"block_%d", block_idx);
    auto ret = bmrt_launch_tensor_ex(encoder->t5_model, name, input_tensors, encoder->net_block0->input_num, output_tensors, encoder->net_block0->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(encoder->handle);
    free(input_tensors);
    free(output_tensors);
    FUNC_TIME_END;
    return 0;
}

inline int t5_encoder_tail(struct t5_encoder *encoder){
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(encoder->net_tail->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(encoder->net_tail->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors, encoder->hidden, encoder->net_tail->input_dtypes[0], encoder->net_tail->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(output_tensors, encoder->hidden, encoder->net_tail->output_dtypes[0], encoder->net_tail->stages[0].output_shapes[0]);
    auto ret = bmrt_launch_tensor_ex(encoder->t5_model, encoder->net_tail->name, input_tensors, encoder->net_tail->input_num, output_tensors, encoder->net_tail->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(encoder->handle);
    free(input_tensors);
    free(output_tensors);
    return 0;
}

int t5_encoder_run(struct t5_encoder *encoder, void* data, void* output){
    FUNC_TIME_START;
    SD3_ASSERT( bm_memcpy_s2d(encoder->handle, encoder->input_tokens, data) == 0 );
    t5_encoder_head(encoder);
    for(int i=0; i<T5_NUM_BLOCKS; i++){
        printf(">>>");
        t5_encoder_block(encoder, i);
    }
    printf("\n");
    t5_encoder_tail(encoder);
    SD3_ASSERT( bm_memcpy_d2s(encoder->handle, output, encoder->hidden) == 0 );
    // int feature_size = 4096;
    // int batch =  77;
    // ops_rmsnorm_inplace( (float*) output, encoder->weight_buffer, batch, feature_size, 1e-6);
    FUNC_TIME_END;
    return 0;
}

int t5_encoder_free(struct t5_encoder *encoder){
    bm_free_device(encoder->handle, encoder->input_tokens);
    bm_free_device(encoder->handle, encoder->hidden);
    bmrt_destroy(encoder->t5_model);
    // bm_dev_free(encoder->handle);
    // free(encoder->weight_buffer);
    free(encoder);
    encoder = NULL;
    return 0;
}


#ifdef __cplusplus
}
#endif