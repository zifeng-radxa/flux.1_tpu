#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

struct flux_dev_device_3 {
    int device_num;
    int device_ids[3];
    void* models[3];
    void* buffer;

    bm_handle_t handles[3];
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

    bm_device_mem_t image_rotary_emb1;
    bm_device_mem_t huge_hidden_states1;
    bm_device_mem_t hidden_states1;
    bm_device_mem_t encoder_hidden_states1;
    bm_device_mem_t temb1;

    bm_device_mem_t image_rotary_emb2;
    bm_device_mem_t huge_hidden_states2;
    bm_device_mem_t hidden_states2;
    bm_device_mem_t encoder_hidden_states2;
    bm_device_mem_t temb2;
    
    bm_device_mem_t predict_noise;
    const bm_net_info_t* net_head;
    const bm_net_info_t* net_transform_block;
    const bm_net_info_t* net_simple_block;
    const bm_net_info_t* net_tail;
};


struct flux_dev_device_3 * flux_dev_multi_device_init(const char** filename, int* device_ids);
int flux_dev_multi_device_run(struct  flux_dev_device_3 *flux_dev, void* input0, void* input1, void* input11, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack);
int flux_dev_multi_device_free(struct flux_dev_device_3 *flux_dev);

struct flux_dev_device_3 * flux_dev_multi_device_init(const char** filename, int* device_ids)
{
    struct flux_dev_device_3 *flux_dev = (struct flux_dev_device_3 *)calloc(1, sizeof(struct flux_dev_device_3));
    flux_dev->device_num = 3;
    loop(i,3){
        printf("device_id: %d, filename: %s\n", device_ids[i], filename[i]);
        flux_dev->device_ids[i] = device_ids[i];
        flux_dev->handles[i]    = get_handle(device_ids[i]);
        flux_dev->models[i]     = bmrt_create(flux_dev->handles[i]);
        SD3_ASSERT(bmrt_load_bmodel(flux_dev->models[i], filename[i])==true);
    }
    flux_dev->net_head             = bmrt_get_network_info(flux_dev->models[0], "dev_head");
    flux_dev->net_transform_block  = bmrt_get_network_info(flux_dev->models[0], "dev_trans_block_0");
    flux_dev->net_simple_block     = bmrt_get_network_info(flux_dev->models[1], "dev_single_trans_block_0");
    flux_dev->net_tail             = bmrt_get_network_info(flux_dev->models[2], "dev_tail");
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[0], &flux_dev->init_states,                 flux_dev->net_head->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[0], &flux_dev->timestep,                    flux_dev->net_head->max_input_bytes[1] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[0], &flux_dev->guidance,                    flux_dev->net_head->max_input_bytes[2] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[0], &flux_dev->pooled_projections,          flux_dev->net_head->max_input_bytes[3] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[0], &flux_dev->init_encoder_hidden_states,  flux_dev->net_head->max_input_bytes[4] ) == 0 );
    // multi device
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[0], &flux_dev->huge_hidden_states,          flux_dev->net_simple_block->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[1], &flux_dev->huge_hidden_states1,         flux_dev->net_simple_block->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[2], &flux_dev->huge_hidden_states2,         flux_dev->net_simple_block->max_input_bytes[0] ) == 0 );

    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[0], &flux_dev->temb,                        flux_dev->net_transform_block->max_input_bytes[2]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[1], &flux_dev->temb1,                       flux_dev->net_transform_block->max_input_bytes[2]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[2], &flux_dev->temb2,                       flux_dev->net_transform_block->max_input_bytes[2]) == 0 );

    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[0], &flux_dev->image_rotary_emb,            flux_dev->net_transform_block->max_input_bytes[3]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[1], &flux_dev->image_rotary_emb1,           flux_dev->net_transform_block->max_input_bytes[3]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[2], &flux_dev->image_rotary_emb2,           flux_dev->net_transform_block->max_input_bytes[3]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_dev->handles[2], &flux_dev->predict_noise, flux_dev->net_tail->max_output_bytes[0]) == 0 );
    // device 0
    bm_device_mem_offset(&flux_dev->huge_hidden_states, &flux_dev->encoder_hidden_states, 0, flux_dev->net_transform_block->max_output_bytes[0]);
    bm_device_mem_offset(&flux_dev->huge_hidden_states, &flux_dev->hidden_states,         flux_dev->net_transform_block->max_output_bytes[0], flux_dev->net_transform_block->max_output_bytes[1]);

    // device 1
    bm_device_mem_offset(&flux_dev->huge_hidden_states1, &flux_dev->encoder_hidden_states1, 0, flux_dev->net_transform_block->max_output_bytes[0]);
    bm_device_mem_offset(&flux_dev->huge_hidden_states1, &flux_dev->hidden_states1,         flux_dev->net_transform_block->max_output_bytes[0], flux_dev->net_transform_block->max_output_bytes[1]);

    // device 2
    bm_device_mem_offset(&flux_dev->huge_hidden_states2, &flux_dev->encoder_hidden_states2, 0, flux_dev->net_transform_block->max_output_bytes[0]);
    bm_device_mem_offset(&flux_dev->huge_hidden_states2, &flux_dev->hidden_states2,         flux_dev->net_transform_block->max_output_bytes[0], flux_dev->net_transform_block->max_output_bytes[1]);

    // flux_dev->buffer = (void*)malloc(flux_dev->net_tail->max_output_bytes[0]);
    return flux_dev;
}

int flux_dev_multi_device_run(struct  flux_dev_device_3 *flux_dev, void* input0, void* input1, void* input11, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack)
{
    FUNC_TIME_START;
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(flux_dev->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(flux_dev->net_head->output_num, sizeof(bm_tensor_t));
    // starts with device 0
    bmrt_tensor_with_device(input_tensors,   flux_dev->init_states,               flux_dev->net_head->input_dtypes[0], flux_dev->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, flux_dev->timestep,                  flux_dev->net_head->input_dtypes[1], flux_dev->net_head->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(input_tensors+2, flux_dev->guidance,                  flux_dev->net_head->input_dtypes[2], flux_dev->net_head->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(input_tensors+3, flux_dev->pooled_projections,        flux_dev->net_head->input_dtypes[3], flux_dev->net_head->stages[0].input_shapes[3]);
    bmrt_tensor_with_device(input_tensors+4, flux_dev->init_encoder_hidden_states,flux_dev->net_head->input_dtypes[4], flux_dev->net_head->stages[0].input_shapes[4]);
    // output  s2d
    bm_memcpy_s2d(flux_dev->handles[0], flux_dev->init_states, input0);
    bm_memcpy_s2d(flux_dev->handles[0], flux_dev->timestep, input1);
    bm_memcpy_s2d(flux_dev->handles[0], flux_dev->guidance, input11);
    bm_memcpy_s2d(flux_dev->handles[0], flux_dev->pooled_projections, input2);
    bm_memcpy_s2d(flux_dev->handles[0], flux_dev->init_encoder_hidden_states, input3);
    bmrt_tensor_with_device(output_tensors,   flux_dev->temb, flux_dev->net_head->output_dtypes[0],                  flux_dev->net_head->stages[0].output_shapes[0]);
    bmrt_tensor_with_device(output_tensors+1, flux_dev->encoder_hidden_states, flux_dev->net_head->output_dtypes[1], flux_dev->net_head->stages[0].output_shapes[1]);
    bmrt_tensor_with_device(output_tensors+2, flux_dev->hidden_states, flux_dev->net_head->output_dtypes[2],         flux_dev->net_head->stages[0].output_shapes[2]);

    {
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_dev->models[0], flux_dev->net_head->name, input_tensors, flux_dev->net_head->input_num, output_tensors, flux_dev->net_head->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_dev->handles[0]);
        DOMAIN_TIME_END("head");
    }

        input_tensors  = (bm_tensor_t*) realloc(input_tensors,  flux_dev->net_transform_block->input_num * sizeof(bm_tensor_t));
        output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_dev->net_transform_block->output_num * sizeof(bm_tensor_t));
        // starts with device 0
        bmrt_tensor_with_device(input_tensors,   flux_dev->hidden_states,               flux_dev->net_transform_block->input_dtypes[0], flux_dev->net_transform_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_dev->encoder_hidden_states,        flux_dev->net_transform_block->input_dtypes[1], flux_dev->net_transform_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_dev->temb,                        flux_dev->net_transform_block->input_dtypes[2], flux_dev->net_transform_block->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(input_tensors+3, flux_dev->image_rotary_emb,            flux_dev->net_transform_block->input_dtypes[3], flux_dev->net_transform_block->stages[0].input_shapes[3]);
        // s2d image_rotary_emb
        SD3_ASSERT(bm_memcpy_s2d(flux_dev->handles[0], flux_dev->image_rotary_emb, rotary_emb) == 0);

        bmrt_tensor_with_device(output_tensors,  flux_dev->encoder_hidden_states,        flux_dev->net_transform_block->output_dtypes[0], flux_dev->net_transform_block->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(output_tensors+1,flux_dev->hidden_states,                flux_dev->net_transform_block->output_dtypes[1], flux_dev->net_transform_block->stages[0].output_shapes[1]);
        // block
        char name[100];
        loop(i, 13){
            DOMAIN_TIME_START;
            sprintf(name, "dev_trans_block_%d", i);
            auto ret = bmrt_launch_tensor_ex(flux_dev->models[0], name, input_tensors, flux_dev->net_transform_block->input_num, output_tensors, flux_dev->net_transform_block->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_dev->handles[0]);
            DOMAIN_TIME_END(name);
            name[0] = '\0';
        }
        float* huge_hidden = (float*)malloc(flux_dev->net_simple_block->max_output_bytes[0]);
        float* temb_data   = (float*)malloc(flux_dev->temb.size);
        {
            DOMAIN_TIME_START;
            // starts with device 1
            
            bm_memcpy_d2s(flux_dev->handles[0], huge_hidden, flux_dev->huge_hidden_states);
            bm_memcpy_s2d(flux_dev->handles[1], flux_dev->huge_hidden_states1, huge_hidden);
            // temp
            bm_memcpy_d2s(flux_dev->handles[0], temb_data, flux_dev->temb);
            bm_memcpy_s2d(flux_dev->handles[1], flux_dev->temb1, temb_data);
            // image_rotary_emb
            bm_memcpy_s2d(flux_dev->handles[1], flux_dev->image_rotary_emb1, rotary_emb);
            DOMAIN_TIME_END("device s2d"); 
        }
        bmrt_tensor_with_device(input_tensors,   flux_dev->hidden_states1,               flux_dev->net_transform_block->input_dtypes[0], flux_dev->net_transform_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_dev->encoder_hidden_states1,        flux_dev->net_transform_block->input_dtypes[1], flux_dev->net_transform_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_dev->temb1,                        flux_dev->net_transform_block->input_dtypes[2], flux_dev->net_transform_block->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(input_tensors+3, flux_dev->image_rotary_emb1,            flux_dev->net_transform_block->input_dtypes[3], flux_dev->net_transform_block->stages[0].input_shapes[3]);
        // output
        bmrt_tensor_with_device(output_tensors,  flux_dev->encoder_hidden_states1,        flux_dev->net_transform_block->output_dtypes[0], flux_dev->net_transform_block->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(output_tensors+1,flux_dev->hidden_states1,                flux_dev->net_transform_block->output_dtypes[1], flux_dev->net_transform_block->stages[0].output_shapes[1]);
        for(int i = 13; i < 19; i++){
            DOMAIN_TIME_START;
            sprintf(name, "dev_trans_block_%d", i);
            auto ret = bmrt_launch_tensor_ex(flux_dev->models[1], name, input_tensors, flux_dev->net_transform_block->input_num, output_tensors, flux_dev->net_transform_block->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_dev->handles[1]);
            DOMAIN_TIME_END(name);
            name[0] = '\0';
        }

        input_tensors  = (bm_tensor_t*) realloc(input_tensors,  flux_dev->net_simple_block->input_num * sizeof(bm_tensor_t));
        output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_dev->net_simple_block->output_num * sizeof(bm_tensor_t));
        bmrt_tensor_with_device(input_tensors,   flux_dev->huge_hidden_states1, flux_dev->net_simple_block->input_dtypes[0], flux_dev->net_simple_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_dev->temb1,               flux_dev->net_simple_block->input_dtypes[1], flux_dev->net_simple_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_dev->image_rotary_emb1,   flux_dev->net_simple_block->input_dtypes[2], flux_dev->net_simple_block->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(output_tensors,  flux_dev->huge_hidden_states1, flux_dev->net_simple_block->output_dtypes[0], flux_dev->net_simple_block->stages[0].output_shapes[0]);
        loop(i, 28){
            DOMAIN_TIME_START;
            sprintf(name, "dev_single_trans_block_%d", i);
            auto ret = bmrt_launch_tensor_ex(flux_dev->models[1], name, input_tensors, flux_dev->net_simple_block->input_num, output_tensors, flux_dev->net_simple_block->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_dev->handles[1]);
            DOMAIN_TIME_END(name);
            name[0] = '\0';
        }
        {
            DOMAIN_TIME_START;
            // starts with device 2
            bm_memcpy_d2s(flux_dev->handles[1], huge_hidden, flux_dev->huge_hidden_states1);
            bm_memcpy_s2d(flux_dev->handles[2], flux_dev->huge_hidden_states2, huge_hidden);
            free(huge_hidden);
            // temp
            bm_memcpy_d2s(flux_dev->handles[1], temb_data, flux_dev->temb1);
            bm_memcpy_s2d(flux_dev->handles[2], flux_dev->temb2, temb_data);
            free(temb_data);
            // image_rotary_emb
            bm_memcpy_s2d(flux_dev->handles[2], flux_dev->image_rotary_emb2, rotary_emb);
            DOMAIN_TIME_END("device s2d");
        }

        bmrt_tensor_with_device(input_tensors,   flux_dev->huge_hidden_states2,          flux_dev->net_simple_block->input_dtypes[0], flux_dev->net_simple_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_dev->temb2,                        flux_dev->net_simple_block->input_dtypes[1], flux_dev->net_simple_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_dev->image_rotary_emb2,            flux_dev->net_simple_block->input_dtypes[2], flux_dev->net_simple_block->stages[0].input_shapes[2]);
        // output
        bmrt_tensor_with_device(output_tensors,  flux_dev->huge_hidden_states2,          flux_dev->net_simple_block->output_dtypes[0], flux_dev->net_simple_block->stages[0].output_shapes[0]);
        for(int i = 28; i < 38; i++){
            DOMAIN_TIME_START;
            sprintf(name, "dev_single_trans_block_%d", i);
            auto ret = bmrt_launch_tensor_ex(flux_dev->models[2], name, input_tensors, flux_dev->net_simple_block->input_num, output_tensors, flux_dev->net_simple_block->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_dev->handles[2]);
            DOMAIN_TIME_END(name);
            name[0] = '\0';
        }

        // tail bmodel 
        input_tensors  = (bm_tensor_t*) realloc(input_tensors, flux_dev->net_tail->input_num * sizeof(bm_tensor_t));
        output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_dev->net_tail->output_num * sizeof(bm_tensor_t));

        bmrt_tensor_with_device(input_tensors,   flux_dev->hidden_states2, flux_dev->net_tail->input_dtypes[0], flux_dev->net_tail->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_dev->temb2,               flux_dev->net_tail->input_dtypes[1], flux_dev->net_tail->stages[0].input_shapes[1]);        
        bmrt_tensor_with_device(output_tensors,  flux_dev->predict_noise, flux_dev->net_tail->output_dtypes[0], flux_dev->net_tail->stages[0].output_shapes[0]);
        {
            DOMAIN_TIME_START;
            auto ret = bmrt_launch_tensor_ex(flux_dev->models[2], flux_dev->net_tail->name, input_tensors, flux_dev->net_tail->input_num, output_tensors, flux_dev->net_tail->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_dev->handles[2]);
            DOMAIN_TIME_END("flux_dev_tail");
        }
        free(input_tensors);
        free(output_tensors);
        // printf("do_unpack %d\n", do_unpack);
        if(do_unpack==1){
            bm_memcpy_d2s(flux_dev->handles[2], flux_dev->buffer, flux_dev->predict_noise);
            convert_pathify_to_latent(flux_dev->buffer, output, 1, 64, 64, 2, 2, 16, 4);
        }else{
            // printf("do not unpack\n");
            bm_memcpy_d2s(flux_dev->handles[2], output, flux_dev->predict_noise);
        }
        FUNC_TIME_END;
        return 0;
}

int flux_dev_multi_device_free(struct flux_dev_device_3 *flux_dev)
{
    bm_free_device(flux_dev->handles[0], flux_dev->init_states);
    bm_free_device(flux_dev->handles[0], flux_dev->timestep);
    bm_free_device(flux_dev->handles[0], flux_dev->guidance);
    bm_free_device(flux_dev->handles[0], flux_dev->pooled_projections);
    bm_free_device(flux_dev->handles[0], flux_dev->init_encoder_hidden_states);
    bm_free_device(flux_dev->handles[0], flux_dev->huge_hidden_states);
    bm_free_device(flux_dev->handles[1], flux_dev->huge_hidden_states1);
    bm_free_device(flux_dev->handles[2], flux_dev->huge_hidden_states2);
    bm_free_device(flux_dev->handles[0], flux_dev->temb);
    bm_free_device(flux_dev->handles[1], flux_dev->temb1);
    bm_free_device(flux_dev->handles[2], flux_dev->temb2);
    bm_free_device(flux_dev->handles[0], flux_dev->image_rotary_emb);
    bm_free_device(flux_dev->handles[1], flux_dev->image_rotary_emb1);
    bm_free_device(flux_dev->handles[2], flux_dev->image_rotary_emb2);
    bm_free_device(flux_dev->handles[2], flux_dev->predict_noise);
    bmrt_destroy(flux_dev->models[0]);
    bmrt_destroy(flux_dev->models[1]);
    bmrt_destroy(flux_dev->models[2]);
    free(flux_dev);
    flux_dev = NULL;
    return 0;
}

#ifdef __cplusplus
}
#endif