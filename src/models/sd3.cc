// boring coding 
/* TODO
* 1. add text inversion for clip_model
*/
#include "model.h"
#define MMDIT_PREFIX    "mmdit_"
#define MMDIT_NUM_BLOCKS  24
#define MMDIT_RES_DTYPE_LEN 4

#ifdef __cplusplus
extern "C" {
#endif

struct mmdit {
    int device_id;
    void* model;
    void* buffer;
    bm_handle_t handle;
    bm_device_mem_t init_states;
    bm_device_mem_t hidden_states;
    bm_device_mem_t temb;
    bm_device_mem_t init_encoder_hidden_states;
    bm_device_mem_t block_encoder_hidden_states;
    bm_device_mem_t pooled_projections;
    bm_device_mem_t timestep;
    bm_device_mem_t predict_noise;
    const bm_net_info_t* net_head;
    const bm_net_info_t* net_block;
    const bm_net_info_t* net_tail;
};

struct vae_decoder {
    int device_id;
    void* model;
    bm_device_mem_t latent;
    bm_device_mem_t img;
    bm_handle_t handle;
    const bm_net_info_t* net;
};


struct mmdit * mmdit_init(const char* filename, int device_id);
inline int mmdit_head(struct mmdit *mmdit);
inline int mmdit_block(struct mmdit *mmdit, int block_idx);
inline int mmdit_tail(struct mmdit *mmdit);
int mmdit_free(struct mmdit *mmdit);
int mmdit_run(struct mmdit *mmdit, void* data0, void* data1, void* data2, void* data3, void* output);

struct mmdit * mmdit_init(const char* filename, int device_id){
    if(READ_FILE_SPEED_SHOW)
        read_file(filename, 100000);
    struct mmdit *mmdit = (struct mmdit *)calloc(1, sizeof(struct mmdit));
    mmdit->handle       = get_handle(device_id);
    mmdit->device_id    = device_id;
    mmdit->model        = bmrt_create(mmdit->handle);
    SD3_ASSERT(bmrt_load_bmodel(mmdit->model, filename)==true);
    auto num_nets       = bmrt_get_network_number(mmdit->model);
    mmdit->net_head     = bmrt_get_network_info(mmdit->model, MMDIT_PREFIX"head");
    mmdit->net_block    = bmrt_get_network_info(mmdit->model, MMDIT_PREFIX"block_0");
    mmdit->net_tail     = bmrt_get_network_info(mmdit->model, MMDIT_PREFIX"tail");
    SD3_ASSERT( bm_malloc_device_byte(mmdit->handle, &mmdit->init_states,                 mmdit->net_head->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(mmdit->handle, &mmdit->init_encoder_hidden_states,  mmdit->net_head->max_input_bytes[1] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(mmdit->handle, &mmdit->pooled_projections,          mmdit->net_head->max_input_bytes[2] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(mmdit->handle, &mmdit->timestep,                    mmdit->net_head->max_input_bytes[3] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(mmdit->handle, &mmdit->hidden_states,               mmdit->net_head->max_output_bytes[0]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(mmdit->handle, &mmdit->temb,                        mmdit->net_head->max_output_bytes[1]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(mmdit->handle, &mmdit->block_encoder_hidden_states, mmdit->net_head->max_output_bytes[2]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(mmdit->handle, &mmdit->predict_noise,               mmdit->net_tail->max_output_bytes[0]) == 0 );
    mmdit->buffer = (void*)calloc(1, mmdit->net_tail->max_output_bytes[0]);
    return mmdit;
}

inline int mmdit_head(struct mmdit *mmdit){
    // input 3 output 3
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(mmdit->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(mmdit->net_head->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   mmdit->init_states,                 mmdit->net_head->input_dtypes[0],   mmdit->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, mmdit->init_encoder_hidden_states,  mmdit->net_head->input_dtypes[1],   mmdit->net_head->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(input_tensors+2, mmdit->pooled_projections,          mmdit->net_head->input_dtypes[2],   mmdit->net_head->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(input_tensors+3, mmdit->timestep,                    mmdit->net_head->input_dtypes[3],   mmdit->net_head->stages[0].input_shapes[3]);
    bmrt_tensor_with_device(output_tensors,  mmdit->hidden_states,               mmdit->net_head->output_dtypes[0],  mmdit->net_head->stages[0].output_shapes[0]);
    bmrt_tensor_with_device(output_tensors+1,mmdit->temb,                        mmdit->net_head->output_dtypes[1],  mmdit->net_head->stages[0].output_shapes[1]);
    bmrt_tensor_with_device(output_tensors+2,mmdit->block_encoder_hidden_states, mmdit->net_head->output_dtypes[2],  mmdit->net_head->stages[0].output_shapes[2]);
    auto ret = bmrt_launch_tensor_ex(mmdit->model, mmdit->net_head->name, input_tensors, mmdit->net_head->input_num, output_tensors, mmdit->net_head->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(mmdit->handle);
    free(input_tensors);
    free(output_tensors);
    return 0;
}

inline int mmdit_block(struct mmdit *mmdit, int block_idx){
    FUNC_TIME_START;
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(mmdit->net_block->input_num , sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(mmdit->net_block->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors  ,  mmdit->hidden_states,    mmdit->net_block->input_dtypes[0],  mmdit->net_block->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1,  mmdit->temb,   mmdit->net_block->input_dtypes[1],  mmdit->net_block->stages[0].input_shapes[1]);
    // input temb
    bmrt_tensor_with_device(input_tensors+2,  mmdit->block_encoder_hidden_states, mmdit->net_block->input_dtypes[2],  mmdit->net_block->stages[0].input_shapes[2]);
    // hidden_state output 
    if(block_idx == MMDIT_NUM_BLOCKS-1){
        bmrt_tensor_with_device(output_tensors,  mmdit->hidden_states, mmdit->net_block->output_dtypes[1], mmdit->net_block->stages[0].output_shapes[1]);
    }else{
        bmrt_tensor_with_device(output_tensors,  mmdit->block_encoder_hidden_states,mmdit->net_block->output_dtypes[0], mmdit->net_block->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(output_tensors+1,mmdit->hidden_states, mmdit->net_block->output_dtypes[1], mmdit->net_block->stages[0].output_shapes[1]);
    }
    // encoder_hidden_state output
    char name[100];
    int output_num = block_idx < MMDIT_NUM_BLOCKS -1 ? mmdit->net_block->output_num : 1;
    sprintf(name, MMDIT_PREFIX"block_%d", block_idx);
    auto ret = bmrt_launch_tensor_ex(mmdit->model, name, input_tensors, mmdit->net_block->input_num, output_tensors, output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(mmdit->handle);
    free(input_tensors);
    free(output_tensors);
    FUNC_TIME_END;
    return 0;
}

inline int mmdit_tail(struct mmdit *mmdit){
    FUNC_TIME_START;
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(mmdit->net_tail->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(mmdit->net_tail->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,  mmdit->hidden_states, mmdit->net_tail->input_dtypes[0],  mmdit->net_tail->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1,mmdit->temb, mmdit->net_tail->input_dtypes[1],  mmdit->net_tail->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(output_tensors, mmdit->predict_noise, mmdit->net_tail->output_dtypes[0], mmdit->net_tail->stages[0].output_shapes[0]);
    auto ret = bmrt_launch_tensor_ex(mmdit->model, mmdit->net_tail->name, input_tensors, mmdit->net_tail->input_num, output_tensors, mmdit->net_tail->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(mmdit->handle);
    free(input_tensors);
    free(output_tensors);
    FUNC_TIME_END;
    return 0;
}

int mmdit_run(struct mmdit *mmdit, void* data0, void* data1, void* data2, void* data3, void* output){
    FUNC_TIME_START;
    SD3_ASSERT( bm_memcpy_s2d(mmdit->handle, mmdit->init_states,                data0) == 0 );
    SD3_ASSERT( bm_memcpy_s2d(mmdit->handle, mmdit->init_encoder_hidden_states, data1) == 0 );
    SD3_ASSERT( bm_memcpy_s2d(mmdit->handle, mmdit->pooled_projections,         data2) == 0 );
    SD3_ASSERT( bm_memcpy_s2d(mmdit->handle, mmdit->timestep,                   data3) == 0 );
    mmdit_head(mmdit);
    for(int i=0; i<MMDIT_NUM_BLOCKS; i++){
        mmdit_block(mmdit, i);
    }
    mmdit_tail(mmdit);
    SD3_ASSERT( bm_memcpy_d2s(mmdit->handle, mmdit->buffer, mmdit->predict_noise) == 0 );
    int batch = mmdit->net_head->stages[0].output_shapes[0].dims[0];
    convert_pathify_to_latent(mmdit->buffer, output, batch, 64, 64, 2, 2, 16, MMDIT_RES_DTYPE_LEN);
    FUNC_TIME_END;
    return 0;
}

int mmdit_free(struct mmdit *mmdit){
    bm_free_device(mmdit->handle, mmdit->init_states);
    bm_free_device(mmdit->handle, mmdit->init_encoder_hidden_states);
    bm_free_device(mmdit->handle, mmdit->pooled_projections);
    bm_free_device(mmdit->handle, mmdit->timestep);
    bm_free_device(mmdit->handle, mmdit->hidden_states);
    bm_free_device(mmdit->handle, mmdit->temb);
    bm_free_device(mmdit->handle, mmdit->block_encoder_hidden_states);
    bmrt_destroy(mmdit->model);
    bm_dev_free(mmdit->handle);
    free(mmdit->buffer);
    free(mmdit);
    mmdit = NULL;
    return 0;

}


#ifdef __cplusplus
}
#endif
