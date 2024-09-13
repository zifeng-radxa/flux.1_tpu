
#include "model.h"


#define CLIP_L_PREFIX   "clip_l_"
#define CLIP_L_NUM_BLOCKS 32
#define CLIP_G_PREFIX   "clip_g_"
#define CLIP_G_NUM_BLOCKS 12

#ifdef __cplusplus
extern "C" {
#endif

struct clip_pooling {
    int device_id;
    void* model;
    bm_device_mem_t input_tokens;
    bm_device_mem_t hidden;
    bm_device_mem_t pooling_embed;
    bm_handle_t handle;
    const bm_net_info_t* net_head;
    const bm_net_info_t* net_tail;
    const bm_net_info_t* net_block;
};

struct clip_l_encoder {
    int device_id;
    void* model;
    bm_device_mem_t input_tokens;
    bm_device_mem_t hidden;
    bm_device_mem_t pooling_embed;
    bm_handle_t     handle;
    const bm_net_info_t* net_head;
    const bm_net_info_t* net_tail;
    const bm_net_info_t* net_block;
};

struct clip_g_encoder {
    int                  device_id;
    void*                model;
    bm_device_mem_t      input_tokens;
    bm_device_mem_t      hidden;
    bm_device_mem_t      pooling_embed;
    bm_handle_t          handle;
    const bm_net_info_t* net_head;
    const bm_net_info_t* net_tail;
    const bm_net_info_t* net_block;
};


struct clip_pooling * clip_pooling_init(const char* filename, int device_id);
int clip_pooling_run(struct clip_pooling *clip_pooling, void* input_tokens, void* pooling_embed);
int clip_pooling_free(struct clip_pooling *clip_pooling);


struct clip_l_encoder * clip_l_encoder_init(const char* filename, int device_id);
int clip_l_encoder_run(struct clip_l_encoder *encoder, void* input_tokens, int clip_skip, void* prompt_embed, void* pooling_embed);
int clip_l_free(struct clip_l_encoder *encoder);


struct clip_g_encoder * clip_g_encoder_init(const char* filename, int device_id);
int clip_g_encoder_run(struct clip_g_encoder *encoder, void* input_tokens, int clip_skip, void* prompt_embed, void* pooling_embed);
int clip_g_free(struct clip_g_encoder *encoder);

struct clip_pooling * clip_pooling_init(const char* filename, int device_id)
{
    struct clip_pooling *clip_pooling = (struct clip_pooling *)calloc(1, sizeof(struct clip_pooling));
    clip_pooling->handle    = get_handle(device_id);
    clip_pooling->device_id = device_id;
    clip_pooling->model = bmrt_create(clip_pooling->handle);
    auto num_nets     = bmrt_get_network_number(clip_pooling->model);
    DEVICE_ASSERT(bmrt_load_bmodel(clip_pooling->model, filename)==true);
    clip_pooling->net_head  = bmrt_get_network_info(clip_pooling->model, "clip_head");
    clip_pooling->net_block = bmrt_get_network_info(clip_pooling->model, "clip_block_0");
    clip_pooling->net_tail  = bmrt_get_network_info(clip_pooling->model, "clip_tail");
    DEVICE_ASSERT( bm_malloc_device_byte(clip_pooling->handle, &clip_pooling->input_tokens, clip_pooling->net_head->max_input_bytes[0] ) == 0 );
    DEVICE_ASSERT( bm_malloc_device_byte(clip_pooling->handle, &clip_pooling->hidden,       clip_pooling->net_head->max_output_bytes[0]) == 0 );
    DEVICE_ASSERT( bm_malloc_device_byte(clip_pooling->handle, &clip_pooling->pooling_embed, clip_pooling->net_tail->max_output_bytes[0]) == 0 );
    return clip_pooling;
}

int clip_pooling_run(struct clip_pooling *clip_pooling, void* input_tokens, void* pooling_embed)
{
    FUNC_TIME_START;
    DEVICE_ASSERT( bm_memcpy_s2d(clip_pooling->handle, clip_pooling->input_tokens, input_tokens) == 0 );
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(clip_pooling->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(clip_pooling->net_head->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors, clip_pooling->input_tokens, clip_pooling->net_head->input_dtypes[0], clip_pooling->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(output_tensors, clip_pooling->hidden, clip_pooling->net_head->output_dtypes[0], clip_pooling->net_head->stages[0].output_shapes[0]);
    auto ret = bmrt_launch_tensor_ex(clip_pooling->model, clip_pooling->net_head->name, input_tensors, clip_pooling->net_head->input_num, output_tensors, clip_pooling->net_head->output_num, true, false);
    DEVICE_ASSERT(ret);
    bm_thread_sync(clip_pooling->handle);
    for(int i=0; i<12; i++){
        bmrt_tensor_with_device(input_tensors, clip_pooling->hidden, clip_pooling->net_block->input_dtypes[0], clip_pooling->net_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(output_tensors, clip_pooling->hidden, clip_pooling->net_block->output_dtypes[0], clip_pooling->net_block->stages[0].output_shapes[0]);
        char name[100];
        sprintf(name, "clip_block_%d", i);
        auto ret = bmrt_launch_tensor_ex(clip_pooling->model, name, input_tensors, clip_pooling->net_block->input_num, output_tensors, clip_pooling->net_block->output_num, true, false);
        DEVICE_ASSERT(ret);
        bm_thread_sync(clip_pooling->handle);
    }
    input_tensors = (bm_tensor_t*) realloc(input_tensors, clip_pooling->net_tail->input_num * sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   clip_pooling->hidden,       clip_pooling->net_tail->input_dtypes[0],  clip_pooling->net_tail->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, clip_pooling->input_tokens, clip_pooling->net_tail->input_dtypes[1],  clip_pooling->net_tail->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(output_tensors,  clip_pooling->pooling_embed, clip_pooling->net_tail->output_dtypes[0], clip_pooling->net_tail->stages[0].output_shapes[0]);
    ret = bmrt_launch_tensor_ex(clip_pooling->model, clip_pooling->net_tail->name, input_tensors, clip_pooling->net_tail->input_num, output_tensors, clip_pooling->net_tail->output_num, true, false);
    DEVICE_ASSERT(ret);
    bm_thread_sync(clip_pooling->handle);
    DEVICE_ASSERT( bm_memcpy_d2s(clip_pooling->handle, pooling_embed, clip_pooling->pooling_embed) == 0 );
    // print pooling_embed data
    FUNC_TIME_END;  
}

int clip_pooling_free(struct clip_pooling *clip_pooling)
{
    bm_free_device(clip_pooling->handle, clip_pooling->input_tokens);
    bm_free_device(clip_pooling->handle, clip_pooling->hidden);
    bm_free_device(clip_pooling->handle, clip_pooling->pooling_embed);
    free(clip_pooling);
    return 0;
}


int clip_l_free(struct clip_l_encoder *encoder){
    bm_free_device(encoder->handle, encoder->input_tokens);
    bm_free_device(encoder->handle, encoder->hidden);
    bm_free_device(encoder->handle, encoder->pooling_embed);
    bmrt_destroy(encoder->model);
    bm_dev_free(encoder->handle);
    free(encoder);
    encoder = NULL;
    return 0;
}

int clip_g_free(struct clip_g_encoder *encoder){
    bm_free_device(encoder->handle, encoder->input_tokens);
    bm_free_device(encoder->handle, encoder->hidden);
    bm_free_device(encoder->handle, encoder->pooling_embed);
    bmrt_destroy(encoder->model);
    bm_dev_free(encoder->handle);
    free(encoder);
    encoder = NULL;
    return 0;
}

struct clip_g_encoder * clip_g_encoder_init(const char* filename, int device_id){
    struct clip_g_encoder *encoder = (struct clip_g_encoder *)calloc(1, sizeof(struct clip_g_encoder));
    encoder->handle    = get_handle(device_id);
    encoder->device_id = device_id;
    encoder->model = bmrt_create(encoder->handle);
    auto num_nets     = bmrt_get_network_number(encoder->model);
    SD3_ASSERT(bmrt_load_bmodel(encoder->model, filename)==true);
    encoder->net_head  = bmrt_get_network_info(encoder->model, "clip_g_head");
    encoder->net_block = bmrt_get_network_info(encoder->model, "clip_g_block_0");
    encoder->net_tail  = bmrt_get_network_info(encoder->model, "clip_g_tail");
    SD3_ASSERT( bm_malloc_device_byte(encoder->handle, &encoder->input_tokens, encoder->net_head->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(encoder->handle, &encoder->hidden,       encoder->net_head->max_output_bytes[0]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(encoder->handle, &encoder->pooling_embed, encoder->net_tail->max_output_bytes[0]) == 0 );
    return encoder;
}

int clip_g_encoder_run(struct clip_g_encoder *encoder, void* input_tokens, int clip_skip, void* prompt_embed, void* pooling_embed){
    FUNC_TIME_START;
    SD3_ASSERT( bm_memcpy_s2d(encoder->handle, encoder->input_tokens, input_tokens) == 0 );
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(encoder->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(encoder->net_head->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,  encoder->input_tokens, encoder->net_head->input_dtypes[0], encoder->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(output_tensors, encoder->hidden, encoder->net_head->output_dtypes[0], encoder->net_head->stages[0].output_shapes[0]);
    auto ret = bmrt_launch_tensor_ex(encoder->model, encoder->net_head->name, input_tensors, encoder->net_head->input_num, output_tensors, encoder->net_head->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(encoder->handle);
    for(int i=0; i<CLIP_G_NUM_BLOCKS; i++){
        bmrt_tensor_with_device(input_tensors, encoder->hidden, encoder->net_block->input_dtypes[0], encoder->net_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(output_tensors, encoder->hidden, encoder->net_block->output_dtypes[0], encoder->net_block->stages[0].output_shapes[0]);
        char name[100];
        sprintf(name, "clip_g_block_%d", i);
        auto ret = bmrt_launch_tensor_ex(encoder->model, name, input_tensors, encoder->net_block->input_num, output_tensors, encoder->net_block->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(encoder->handle);
        if(i == CLIP_G_NUM_BLOCKS - 2 - clip_skip ){
            SD3_ASSERT( bm_memcpy_d2s(encoder->handle, prompt_embed, encoder->hidden) == 0 );
        }
    }
    input_tensors = (bm_tensor_t*) realloc(input_tensors, encoder->net_tail->input_num * sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   encoder->hidden,       encoder->net_tail->input_dtypes[0],  encoder->net_tail->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, encoder->input_tokens, encoder->net_tail->input_dtypes[1],  encoder->net_tail->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(output_tensors,  encoder->pooling_embed, encoder->net_tail->output_dtypes[0], encoder->net_tail->stages[0].output_shapes[0]);
    ret = bmrt_launch_tensor_ex(encoder->model, encoder->net_tail->name, input_tensors, encoder->net_tail->input_num, output_tensors, encoder->net_tail->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(encoder->handle);
    SD3_ASSERT( bm_memcpy_d2s(encoder->handle, pooling_embed, encoder->pooling_embed) == 0 );
    FUNC_TIME_END;
    free(input_tensors);
    free(output_tensors);
    return 0;
}

struct clip_l_encoder * clip_l_encoder_init(const char* filename, int device_id){
    struct clip_l_encoder *encoder = (struct clip_l_encoder *)calloc(1, sizeof(struct clip_l_encoder));
    encoder->handle    = get_handle(device_id);
    encoder->device_id = device_id;
    encoder->model = bmrt_create(encoder->handle);
    auto num_nets     = bmrt_get_network_number(encoder->model);
    SD3_ASSERT(bmrt_load_bmodel(encoder->model, filename)==true);
    encoder->net_head = bmrt_get_network_info(encoder->model, CLIP_L_PREFIX"head");
    encoder->net_tail = bmrt_get_network_info(encoder->model, CLIP_L_PREFIX"tail");
    encoder->net_block = bmrt_get_network_info(encoder->model, CLIP_L_PREFIX"block_0");
    SD3_ASSERT( bm_malloc_device_byte(encoder->handle, &encoder->input_tokens,  encoder->net_head->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(encoder->handle, &encoder->hidden,        encoder->net_head->max_output_bytes[0]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(encoder->handle, &encoder->pooling_embed, encoder->net_tail->max_output_bytes[0]) == 0 );
    return encoder;
}

int clip_l_encoder_run(struct clip_l_encoder *encoder, void* input_tokens, int clip_skip, void* prompt_embed, void* pooling_embed){
    FUNC_TIME_START;
    SD3_ASSERT( bm_memcpy_s2d(encoder->handle, encoder->input_tokens, input_tokens) == 0 );
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(encoder->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(encoder->net_head->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors, encoder->input_tokens, encoder->net_head->input_dtypes[0], encoder->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(output_tensors, encoder->hidden, encoder->net_head->output_dtypes[0], encoder->net_head->stages[0].output_shapes[0]);
    auto ret = bmrt_launch_tensor_ex(encoder->model, encoder->net_head->name, input_tensors, encoder->net_head->input_num, output_tensors, encoder->net_head->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(encoder->handle);
    for(int i=0; i<CLIP_L_NUM_BLOCKS; i++){
        bmrt_tensor_with_device(input_tensors, encoder->hidden, encoder->net_block->input_dtypes[0], encoder->net_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(output_tensors, encoder->hidden, encoder->net_block->output_dtypes[0], encoder->net_block->stages[0].output_shapes[0]);
        char name[100];
        sprintf(name, CLIP_L_PREFIX"block_%d", i);
        auto ret = bmrt_launch_tensor_ex(encoder->model, name, input_tensors, encoder->net_block->input_num, output_tensors, encoder->net_block->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(encoder->handle);
        if(i == CLIP_L_NUM_BLOCKS - 2 - clip_skip ){
            SD3_ASSERT( bm_memcpy_d2s(encoder->handle, prompt_embed, encoder->hidden) == 0 );
        }
    }
    input_tensors = (bm_tensor_t*) realloc(input_tensors, encoder->net_tail->input_num * sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   encoder->hidden,       encoder->net_tail->input_dtypes[0],  encoder->net_tail->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, encoder->input_tokens, encoder->net_tail->input_dtypes[1],  encoder->net_tail->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(output_tensors,  encoder->pooling_embed, encoder->net_tail->output_dtypes[0], encoder->net_tail->stages[0].output_shapes[0]);
    ret = bmrt_launch_tensor_ex(encoder->model, encoder->net_tail->name, input_tensors, encoder->net_tail->input_num, output_tensors, encoder->net_tail->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(encoder->handle);
    SD3_ASSERT( bm_memcpy_d2s(encoder->handle, pooling_embed, encoder->pooling_embed) == 0 );
    free(input_tensors);
    free(output_tensors);
    FUNC_TIME_END;
    return 0;
}


#ifdef __cplusplus
}
#endif