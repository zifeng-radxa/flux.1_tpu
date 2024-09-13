#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

struct flux_dev {
    int device_id;
    void* model;
    void* buffer;
    bm_handle_t handle;

    bm_device_mem_t init_states;
    bm_device_mem_t timestep;
    bm_device_mem_t guidance;
    bm_device_mem_t pooled_projections;
    bm_device_mem_t init_encoder_hidden_states;
    
    bm_device_mem_t image_rotary_emb;
    bm_device_mem_t huge_hidden_states;
    bm_device_mem_t hidden_states;
    bm_device_mem_t encoder_hidden_states;
    bm_device_mem_t temb;

    bm_device_mem_t predict_noise;
    const bm_net_info_t* net_head;
    const bm_net_info_t* net_transform_block;
    const bm_net_info_t* net_simple_block;
    const bm_net_info_t* net_tail;
};

struct flux_dev * flux_dev_init(const char* filename, int device_id);
int flux_dev_run(struct flux_dev * flux_dev, void* input0, void* input1, void* input11, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack);
int flux_dev_free(struct flux_dev * flux_dev);


struct flux_dev * flux_dev_init(const char* filename, int device_id)
{
    struct flux_dev *flux_dev = (struct flux_dev *)calloc(1, sizeof(struct flux_dev));
    flux_dev->handle                 = get_handle(device_id);
    flux_dev->device_id              = device_id;
    flux_dev->model                  = bmrt_create(flux_dev->handle);
    SD3_ASSERT(bmrt_load_bmodel(flux_dev->model, filename)==true);
    // auto num_nets              = bmrt_get_network_number(flux_dev->model);
    flux_dev->net_head               = bmrt_get_network_info(flux_dev->model, "dev_head");
    flux_dev->net_tail               = bmrt_get_network_info(flux_dev->model, "dev_tail");
    flux_dev->net_transform_block    = bmrt_get_network_info(flux_dev->model, "dev_trans_block_0");
    flux_dev->net_simple_block       = bmrt_get_network_info(flux_dev->model, "dev_single_trans_block_0");
    
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->init_states,                 flux_dev->net_head->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->timestep,                    flux_dev->net_head->max_input_bytes[1] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->guidance,                    flux_dev->net_head->max_input_bytes[2] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->pooled_projections,          flux_dev->net_head->max_input_bytes[3] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->init_encoder_hidden_states,  flux_dev->net_head->max_input_bytes[4] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->huge_hidden_states,          flux_dev->net_simple_block->max_input_bytes[0] ) == 0 );
    
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->temb,                        flux_dev->net_transform_block->max_input_bytes[2]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->image_rotary_emb,            flux_dev->net_transform_block->max_input_bytes[3]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handle, &flux_dev->predict_noise,               flux_dev->net_tail->max_output_bytes[0]) == 0 );
    bm_device_mem_offset(&flux_dev->huge_hidden_states, &flux_dev->encoder_hidden_states, 0, flux_dev->net_transform_block->max_output_bytes[0]);
    bm_device_mem_offset(&flux_dev->huge_hidden_states, &flux_dev->hidden_states,         flux_dev->net_transform_block->max_output_bytes[0], flux_dev->net_transform_block->max_output_bytes[1]);

    return flux_dev;
}

int flux_dev_run(struct flux_dev * flux_dev, void* input0, void* input1, void* input11, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack)
{
    FUNC_TIME_START;
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(flux_dev->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(flux_dev->net_head->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   flux_dev->init_states,                 flux_dev->net_head->input_dtypes[0],   flux_dev->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, flux_dev->timestep,                    flux_dev->net_head->input_dtypes[1],   flux_dev->net_head->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(input_tensors+2, flux_dev->guidance,                    flux_dev->net_head->input_dtypes[2],   flux_dev->net_head->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(input_tensors+3, flux_dev->pooled_projections,          flux_dev->net_head->input_dtypes[3],   flux_dev->net_head->stages[0].input_shapes[3]);
    bmrt_tensor_with_device(input_tensors+4, flux_dev->init_encoder_hidden_states,  flux_dev->net_head->input_dtypes[4],   flux_dev->net_head->stages[0].input_shapes[4]);
    // s2d
    SD3_ASSERT(bm_memcpy_s2d(flux_dev->handle, flux_dev->init_states, input0) == 0);
    SD3_ASSERT(bm_memcpy_s2d(flux_dev->handle, flux_dev->timestep,    input1) == 0);
    SD3_ASSERT(bm_memcpy_s2d(flux_dev->handle, flux_dev->guidance,    input11) == 0);
    SD3_ASSERT(bm_memcpy_s2d(flux_dev->handle, flux_dev->pooled_projections, input2) == 0);
    SD3_ASSERT(bm_memcpy_s2d(flux_dev->handle, flux_dev->init_encoder_hidden_states, input3) == 0);

    bmrt_tensor_with_device(output_tensors,  flux_dev->temb,                        flux_dev->net_head->output_dtypes[0],  flux_dev->net_head->stages[0].output_shapes[0]);
    bmrt_tensor_with_device(output_tensors+1,flux_dev->encoder_hidden_states,       flux_dev->net_head->output_dtypes[1],  flux_dev->net_head->stages[0].output_shapes[1]);
    bmrt_tensor_with_device(output_tensors+2,flux_dev->hidden_states,               flux_dev->net_head->output_dtypes[2],  flux_dev->net_head->stages[0].output_shapes[2]);
    {
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_dev->model, flux_dev->net_head->name, input_tensors, flux_dev->net_head->input_num, output_tensors, flux_dev->net_head->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_dev->handle);
        DOMAIN_TIME_END("flux_dev_head");
    }
        // input: hidden_states, encoder_hidden_states, temb, image_rotary_emb
        // output: encoder_hidden_states, hidden_states
        input_tensors  = (bm_tensor_t*) realloc(input_tensors, flux_dev->net_transform_block->input_num * sizeof(bm_tensor_t));
        output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_dev->net_transform_block->output_num * sizeof(bm_tensor_t));
        bmrt_tensor_with_device(input_tensors,   flux_dev->hidden_states,               flux_dev->net_transform_block->input_dtypes[0], flux_dev->net_transform_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_dev->encoder_hidden_states,        flux_dev->net_transform_block->input_dtypes[1], flux_dev->net_transform_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_dev->temb,                        flux_dev->net_transform_block->input_dtypes[2], flux_dev->net_transform_block->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(input_tensors+3, flux_dev->image_rotary_emb,            flux_dev->net_transform_block->input_dtypes[3], flux_dev->net_transform_block->stages[0].input_shapes[3]);
        // s2d image_rotary_emb
        SD3_ASSERT(bm_memcpy_s2d(flux_dev->handle, flux_dev->image_rotary_emb, rotary_emb) == 0);

        bmrt_tensor_with_device(output_tensors,  flux_dev->encoder_hidden_states,        flux_dev->net_transform_block->output_dtypes[0], flux_dev->net_transform_block->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(output_tensors+1,flux_dev->hidden_states,               flux_dev->net_transform_block->output_dtypes[1], flux_dev->net_transform_block->stages[0].output_shapes[1]);
        char name[100];
    // transform block
    loop(i, 19)
    {
        // name: schnell_transformer_block_0
        sprintf(name, "dev_trans_block_%d", i);
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_dev->model, name, input_tensors, flux_dev->net_transform_block->input_num, output_tensors, flux_dev->net_transform_block->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_dev->handle);
        DOMAIN_TIME_END(name);
        strcpy(name, "");
    }
    // simple transform block
    // input : huge_hidden_states, temb, image_rotary_emb
    // output: huge_hidden_states
    // name  : schnell_single_transformer_block_0
    input_tensors  = (bm_tensor_t*) realloc(input_tensors, flux_dev->net_simple_block->input_num * sizeof(bm_tensor_t));
    output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_dev->net_simple_block->output_num * sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   flux_dev->huge_hidden_states,          flux_dev->net_simple_block->input_dtypes[0], flux_dev->net_simple_block->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, flux_dev->temb,                        flux_dev->net_simple_block->input_dtypes[1], flux_dev->net_simple_block->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(input_tensors+2, flux_dev->image_rotary_emb,            flux_dev->net_simple_block->input_dtypes[2], flux_dev->net_simple_block->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(output_tensors,  flux_dev->huge_hidden_states,          flux_dev->net_simple_block->output_dtypes[0], flux_dev->net_simple_block->stages[0].output_shapes[0]);
    
    loop(i, 38)
    {
        sprintf(name, "dev_single_trans_block_%d", i);
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_dev->model, name, input_tensors, flux_dev->net_simple_block->input_num, output_tensors, flux_dev->net_simple_block->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_dev->handle);
        DOMAIN_TIME_END(name);
        strcpy(name, "");
    }
    // tail
    // input : hidden_states, temb
    // output: hidden_states
    // name  : schnell_tail
    input_tensors  = (bm_tensor_t*) realloc(input_tensors, flux_dev->net_tail->input_num * sizeof(bm_tensor_t));
    output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_dev->net_tail->output_num * sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   flux_dev->hidden_states,               flux_dev->net_tail->input_dtypes[0], flux_dev->net_tail->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, flux_dev->temb,                        flux_dev->net_tail->input_dtypes[1], flux_dev->net_tail->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(output_tensors,  flux_dev->predict_noise,               flux_dev->net_tail->output_dtypes[0], flux_dev->net_tail->stages[0].output_shapes[0]);
    {
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_dev->model, flux_dev->net_tail->name, input_tensors, flux_dev->net_tail->input_num, output_tensors, flux_dev->net_tail->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_dev->handle);
        DOMAIN_TIME_END("flux_dev_tail");
    }
    free(input_tensors);
    free(output_tensors);
    // d2s
    // printf("do_unpack %d\n", do_unpack);
    if(do_unpack == 1)
    {
        SD3_ASSERT(bm_memcpy_d2s(flux_dev->handle, flux_dev->buffer, flux_dev->predict_noise) == 0);
        convert_pathify_to_latent(flux_dev->buffer, output, 1, 64, 64, 2, 2, 16, 4);
    }else{
        // printf("do not unpack\n");
        SD3_ASSERT(bm_memcpy_d2s(flux_dev->handle, output, flux_dev->predict_noise) == 0);
    }
    FUNC_TIME_END;
    return 0;
}


int flux_dev_free(struct flux_dev *flux_dev)
{
    bm_free_device(flux_dev->handle, flux_dev->init_states);
    bm_free_device(flux_dev->handle, flux_dev->timestep);
    bm_free_device(flux_dev->handle, flux_dev->guidance);
    bm_free_device(flux_dev->handle, flux_dev->pooled_projections);
    bm_free_device(flux_dev->handle, flux_dev->init_encoder_hidden_states);
    bm_free_device(flux_dev->handle, flux_dev->image_rotary_emb);
    bm_free_device(flux_dev->handle, flux_dev->huge_hidden_states);
    bm_free_device(flux_dev->handle, flux_dev->temb);
    bm_free_device(flux_dev->handle, flux_dev->predict_noise);
    bmrt_destroy(flux_dev->model);
    free(flux_dev);
    flux_dev = NULL;
    return 0;
}

#ifdef __cplusplus
}
#endif