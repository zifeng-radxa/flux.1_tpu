#include "model.h"
// flux transformer
#define FLUX_SCHNELL_PREFIX "schnell_"
#define FLUX_SCHNELL_SIMPLE_NUM_BLOCKS 38
#define FLUX_SCHNELL_NUM_BLOCKS 19

#ifdef __cplusplus
extern "C" {
#endif

struct flux_schnell {
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


struct flux_schnell_device_3 {
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


struct flux_schnell * flux_schnell_init(const char* filename, int device_id);
int flux_schnell_run(struct flux_schnell *flux_schnell, void* input0, void* input1, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack);
int flux_schnell_free(struct flux_schnell *flux_schnell);

struct flux_schnell_device_3 * flux_schnell_multi_device_init(const char** filename, int* device_ids);
int flux_schnell_multi_device_run(struct flux_schnell_device_3 *flux_schnell, void* input0, void* input1, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack);
int flux_schnell_multi_device_free(struct flux_schnell_device_3 *flux_schnell);

struct flux_schnell_device_3 * flux_schnell_multi_device_init(const char** filename, int* device_ids)
{
    struct flux_schnell_device_3 *flux_schnell = (struct flux_schnell_device_3 *)calloc(1, sizeof(struct flux_schnell_device_3));
    flux_schnell->device_num = 3;
    loop(i,3){
        printf("device_id: %d, filename: %s\n", device_ids[i], filename[i]);
        flux_schnell->device_ids[i] = device_ids[i];
        flux_schnell->handles[i]    = get_handle(device_ids[i]);
        flux_schnell->models[i]     = bmrt_create(flux_schnell->handles[i]);
        SD3_ASSERT(bmrt_load_bmodel(flux_schnell->models[i], filename[i])==true);
    }
    flux_schnell->net_head             = bmrt_get_network_info(flux_schnell->models[0], "schnell_head");
    flux_schnell->net_transform_block  = bmrt_get_network_info(flux_schnell->models[0], "schnell_transformer_block_0");
    flux_schnell->net_simple_block     = bmrt_get_network_info(flux_schnell->models[1], "schnell_single_transformer_block_0");
    flux_schnell->net_tail             = bmrt_get_network_info(flux_schnell->models[2], "schnell_tail");
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[0], &flux_schnell->init_states,                 flux_schnell->net_head->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[0], &flux_schnell->timestep,                    flux_schnell->net_head->max_input_bytes[1] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[0], &flux_schnell->guidance,                    flux_schnell->net_head->max_input_bytes[1] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[0], &flux_schnell->pooled_projections,          flux_schnell->net_head->max_input_bytes[2] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[0], &flux_schnell->init_encoder_hidden_states,  flux_schnell->net_head->max_input_bytes[3] ) == 0 );
    // multi device
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[0], &flux_schnell->huge_hidden_states,          flux_schnell->net_simple_block->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[1], &flux_schnell->huge_hidden_states1,         flux_schnell->net_simple_block->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[2], &flux_schnell->huge_hidden_states2,         flux_schnell->net_simple_block->max_input_bytes[0] ) == 0 );

    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[0], &flux_schnell->temb,                        flux_schnell->net_transform_block->max_input_bytes[2]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[1], &flux_schnell->temb1,                       flux_schnell->net_transform_block->max_input_bytes[2]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[2], &flux_schnell->temb2,                       flux_schnell->net_transform_block->max_input_bytes[2]) == 0 );

    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[0], &flux_schnell->image_rotary_emb,            flux_schnell->net_transform_block->max_input_bytes[3]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[1], &flux_schnell->image_rotary_emb1,           flux_schnell->net_transform_block->max_input_bytes[3]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[2], &flux_schnell->image_rotary_emb2,           flux_schnell->net_transform_block->max_input_bytes[3]) == 0 );
    // device 0
    bm_device_mem_offset(&flux_schnell->huge_hidden_states, &flux_schnell->encoder_hidden_states, 0, flux_schnell->net_transform_block->max_output_bytes[0]);
    bm_device_mem_offset(&flux_schnell->huge_hidden_states, &flux_schnell->hidden_states,         flux_schnell->net_transform_block->max_output_bytes[0], flux_schnell->net_transform_block->max_output_bytes[1]);

    // device 1
    bm_device_mem_offset(&flux_schnell->huge_hidden_states1, &flux_schnell->encoder_hidden_states1, 0, flux_schnell->net_transform_block->max_output_bytes[0]);
    bm_device_mem_offset(&flux_schnell->huge_hidden_states1, &flux_schnell->hidden_states1,         flux_schnell->net_transform_block->max_output_bytes[0], flux_schnell->net_transform_block->max_output_bytes[1]);

    // device 2
    bm_device_mem_offset(&flux_schnell->huge_hidden_states2, &flux_schnell->encoder_hidden_states2, 0, flux_schnell->net_transform_block->max_output_bytes[0]);
    bm_device_mem_offset(&flux_schnell->huge_hidden_states2, &flux_schnell->hidden_states2,         flux_schnell->net_transform_block->max_output_bytes[0], flux_schnell->net_transform_block->max_output_bytes[1]);

    // predict noise
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handles[2], &flux_schnell->predict_noise, flux_schnell->net_tail->max_output_bytes[0]) == 0 );
    // flux_schnell->buffer = (void*)malloc(flux_schnell->net_tail->max_output_bytes[0]);
    return flux_schnell;
}

int flux_schnell_multi_device_run(struct flux_schnell_device_3 *flux_schnell, void* input0, void* input1, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack)
{
    FUNC_TIME_START;
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(flux_schnell->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(flux_schnell->net_head->output_num, sizeof(bm_tensor_t));
    // starts with device 0
    bmrt_tensor_with_device(input_tensors,   flux_schnell->init_states,  flux_schnell->net_head->input_dtypes[0], flux_schnell->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, flux_schnell->timestep,    flux_schnell->net_head->input_dtypes[1], flux_schnell->net_head->stages[0].input_shapes[1]);
    // bmrt_tensor_with_device(input_tensors+2, flux_schnell->guidance,    flux_schnell->net_head->input_dtypes[2], flux_schnell->net_head->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(input_tensors+2, flux_schnell->pooled_projections, flux_schnell->net_head->input_dtypes[2], flux_schnell->net_head->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(input_tensors+3, flux_schnell->init_encoder_hidden_states, flux_schnell->net_head->input_dtypes[3], flux_schnell->net_head->stages[0].input_shapes[3]);
    // output  s2d
    bm_memcpy_s2d(flux_schnell->handles[0], flux_schnell->init_states, input0);
    bm_memcpy_s2d(flux_schnell->handles[0], flux_schnell->timestep, input1);
    bm_memcpy_s2d(flux_schnell->handles[0], flux_schnell->pooled_projections, input2);
    bm_memcpy_s2d(flux_schnell->handles[0], flux_schnell->init_encoder_hidden_states, input3);
    bmrt_tensor_with_device(output_tensors,   flux_schnell->temb, flux_schnell->net_head->output_dtypes[0],                  flux_schnell->net_head->stages[0].output_shapes[0]);
    bmrt_tensor_with_device(output_tensors+1, flux_schnell->encoder_hidden_states, flux_schnell->net_head->output_dtypes[1], flux_schnell->net_head->stages[0].output_shapes[1]);
    bmrt_tensor_with_device(output_tensors+2, flux_schnell->hidden_states, flux_schnell->net_head->output_dtypes[2],         flux_schnell->net_head->stages[0].output_shapes[2]);

    {
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_schnell->models[0], flux_schnell->net_head->name, input_tensors, flux_schnell->net_head->input_num, output_tensors, flux_schnell->net_head->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_schnell->handles[0]);
        DOMAIN_TIME_END("head");
    }

        input_tensors  = (bm_tensor_t*) realloc(input_tensors,  flux_schnell->net_transform_block->input_num * sizeof(bm_tensor_t));
        output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_schnell->net_transform_block->output_num * sizeof(bm_tensor_t));
        // starts with device 0
        bmrt_tensor_with_device(input_tensors,   flux_schnell->hidden_states,               flux_schnell->net_transform_block->input_dtypes[0], flux_schnell->net_transform_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_schnell->encoder_hidden_states,        flux_schnell->net_transform_block->input_dtypes[1], flux_schnell->net_transform_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_schnell->temb,                        flux_schnell->net_transform_block->input_dtypes[2], flux_schnell->net_transform_block->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(input_tensors+3, flux_schnell->image_rotary_emb,            flux_schnell->net_transform_block->input_dtypes[3], flux_schnell->net_transform_block->stages[0].input_shapes[3]);
        // s2d image_rotary_emb
        SD3_ASSERT(bm_memcpy_s2d(flux_schnell->handles[0], flux_schnell->image_rotary_emb, rotary_emb) == 0);

        bmrt_tensor_with_device(output_tensors,  flux_schnell->encoder_hidden_states,        flux_schnell->net_transform_block->output_dtypes[0], flux_schnell->net_transform_block->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(output_tensors+1,flux_schnell->hidden_states,                flux_schnell->net_transform_block->output_dtypes[1], flux_schnell->net_transform_block->stages[0].output_shapes[1]);
        // block
        char name[100];
        loop(i, 13){
            DOMAIN_TIME_START;
            sprintf(name, "schnell_transformer_block_%d", i);
            auto ret = bmrt_launch_tensor_ex(flux_schnell->models[0], name, input_tensors, flux_schnell->net_transform_block->input_num, output_tensors, flux_schnell->net_transform_block->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_schnell->handles[0]);
            DOMAIN_TIME_END(name);
            name[0] = '\0';
        }
        float* huge_hidden = (float*)malloc(flux_schnell->net_simple_block->max_output_bytes[0]);
        float* temb_data   = (float*)malloc(flux_schnell->temb.size);
        {
            DOMAIN_TIME_START;
            // starts with device 1
            
            bm_memcpy_d2s(flux_schnell->handles[0], huge_hidden, flux_schnell->huge_hidden_states);
            bm_memcpy_s2d(flux_schnell->handles[1], flux_schnell->huge_hidden_states1, huge_hidden);
            // temp
            bm_memcpy_d2s(flux_schnell->handles[0], temb_data, flux_schnell->temb);
            bm_memcpy_s2d(flux_schnell->handles[1], flux_schnell->temb1, temb_data);
            // image_rotary_emb
            bm_memcpy_s2d(flux_schnell->handles[1], flux_schnell->image_rotary_emb1, rotary_emb);
            DOMAIN_TIME_END("device s2d"); 
        }
        bmrt_tensor_with_device(input_tensors,   flux_schnell->hidden_states1,               flux_schnell->net_transform_block->input_dtypes[0], flux_schnell->net_transform_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_schnell->encoder_hidden_states1,        flux_schnell->net_transform_block->input_dtypes[1], flux_schnell->net_transform_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_schnell->temb1,                        flux_schnell->net_transform_block->input_dtypes[2], flux_schnell->net_transform_block->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(input_tensors+3, flux_schnell->image_rotary_emb1,            flux_schnell->net_transform_block->input_dtypes[3], flux_schnell->net_transform_block->stages[0].input_shapes[3]);
        // output
        bmrt_tensor_with_device(output_tensors,  flux_schnell->encoder_hidden_states1,        flux_schnell->net_transform_block->output_dtypes[0], flux_schnell->net_transform_block->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(output_tensors+1,flux_schnell->hidden_states1,                flux_schnell->net_transform_block->output_dtypes[1], flux_schnell->net_transform_block->stages[0].output_shapes[1]);
        for(int i = 13; i < 19; i++){
            DOMAIN_TIME_START;
            sprintf(name, "schnell_transformer_block_%d", i);
            auto ret = bmrt_launch_tensor_ex(flux_schnell->models[1], name, input_tensors, flux_schnell->net_transform_block->input_num, output_tensors, flux_schnell->net_transform_block->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_schnell->handles[1]);
            DOMAIN_TIME_END(name);
            name[0] = '\0';
        }

        input_tensors  = (bm_tensor_t*) realloc(input_tensors,  flux_schnell->net_simple_block->input_num * sizeof(bm_tensor_t));
        output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_schnell->net_simple_block->output_num * sizeof(bm_tensor_t));
        bmrt_tensor_with_device(input_tensors,   flux_schnell->huge_hidden_states1, flux_schnell->net_simple_block->input_dtypes[0], flux_schnell->net_simple_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_schnell->temb1,               flux_schnell->net_simple_block->input_dtypes[1], flux_schnell->net_simple_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_schnell->image_rotary_emb1,   flux_schnell->net_simple_block->input_dtypes[2], flux_schnell->net_simple_block->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(output_tensors,  flux_schnell->huge_hidden_states1, flux_schnell->net_simple_block->output_dtypes[0], flux_schnell->net_simple_block->stages[0].output_shapes[0]);
        loop(i, 28){
            DOMAIN_TIME_START;
            sprintf(name, "schnell_single_transformer_block_%d", i);
            auto ret = bmrt_launch_tensor_ex(flux_schnell->models[1], name, input_tensors, flux_schnell->net_simple_block->input_num, output_tensors, flux_schnell->net_simple_block->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_schnell->handles[1]);
            DOMAIN_TIME_END(name);
            name[0] = '\0';
        }
        {
            DOMAIN_TIME_START;
            // starts with device 2
            bm_memcpy_d2s(flux_schnell->handles[1], huge_hidden, flux_schnell->huge_hidden_states1);
            bm_memcpy_s2d(flux_schnell->handles[2], flux_schnell->huge_hidden_states2, huge_hidden);
            free(huge_hidden);
            // temp
            bm_memcpy_d2s(flux_schnell->handles[1], temb_data, flux_schnell->temb1);
            bm_memcpy_s2d(flux_schnell->handles[2], flux_schnell->temb2, temb_data);
            free(temb_data);
            // image_rotary_emb
            bm_memcpy_s2d(flux_schnell->handles[2], flux_schnell->image_rotary_emb2, rotary_emb);
            DOMAIN_TIME_END("device s2d");
        }

        bmrt_tensor_with_device(input_tensors,   flux_schnell->huge_hidden_states2,          flux_schnell->net_simple_block->input_dtypes[0], flux_schnell->net_simple_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_schnell->temb2,                        flux_schnell->net_simple_block->input_dtypes[1], flux_schnell->net_simple_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_schnell->image_rotary_emb2,            flux_schnell->net_simple_block->input_dtypes[2], flux_schnell->net_simple_block->stages[0].input_shapes[2]);
        // output
        bmrt_tensor_with_device(output_tensors,  flux_schnell->huge_hidden_states2,          flux_schnell->net_simple_block->output_dtypes[0], flux_schnell->net_simple_block->stages[0].output_shapes[0]);
        for(int i = 28; i < 38; i++){
            DOMAIN_TIME_START;
            sprintf(name, "schnell_single_transformer_block_%d", i);
            auto ret = bmrt_launch_tensor_ex(flux_schnell->models[2], name, input_tensors, flux_schnell->net_simple_block->input_num, output_tensors, flux_schnell->net_simple_block->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_schnell->handles[2]);
            DOMAIN_TIME_END(name);
            name[0] = '\0';
        }

        // tail bmodel 
        input_tensors  = (bm_tensor_t*) realloc(input_tensors, flux_schnell->net_tail->input_num * sizeof(bm_tensor_t));
        output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_schnell->net_tail->output_num * sizeof(bm_tensor_t));

        bmrt_tensor_with_device(input_tensors,   flux_schnell->hidden_states2, flux_schnell->net_tail->input_dtypes[0], flux_schnell->net_tail->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_schnell->temb2,               flux_schnell->net_tail->input_dtypes[1], flux_schnell->net_tail->stages[0].input_shapes[1]);        
        bmrt_tensor_with_device(output_tensors,  flux_schnell->predict_noise, flux_schnell->net_tail->output_dtypes[0], flux_schnell->net_tail->stages[0].output_shapes[0]);
        {
            DOMAIN_TIME_START;
            auto ret = bmrt_launch_tensor_ex(flux_schnell->models[2], flux_schnell->net_tail->name, input_tensors, flux_schnell->net_tail->input_num, output_tensors, flux_schnell->net_tail->output_num, true, false);
            SD3_ASSERT(ret);
            bm_thread_sync(flux_schnell->handles[2]);
            DOMAIN_TIME_END("flux_schnell_tail");
        }
        free(input_tensors);
        free(output_tensors);
        // printf("do_unpack %d\n", do_unpack);
        if(do_unpack==1){
            bm_memcpy_d2s(flux_schnell->handles[2], flux_schnell->buffer, flux_schnell->predict_noise);
            convert_pathify_to_latent(flux_schnell->buffer, output, 1, 64, 64, 2, 2, 16, 4);
        }else{
            // printf("do not unpack\n");
            bm_memcpy_d2s(flux_schnell->handles[2], output, flux_schnell->predict_noise);
        }
        FUNC_TIME_END;
        return 0;
}

int flux_schnell_multi_device_free(struct flux_schnell_device_3 *flux_schnell)
{
    bm_free_device(flux_schnell->handles[0], flux_schnell->init_states);
    bm_free_device(flux_schnell->handles[0], flux_schnell->timestep);
    bm_free_device(flux_schnell->handles[0], flux_schnell->guidance);
    bm_free_device(flux_schnell->handles[0], flux_schnell->pooled_projections);
    bm_free_device(flux_schnell->handles[0], flux_schnell->init_encoder_hidden_states);
    bm_free_device(flux_schnell->handles[0], flux_schnell->huge_hidden_states);
    bm_free_device(flux_schnell->handles[1], flux_schnell->huge_hidden_states1);
    bm_free_device(flux_schnell->handles[2], flux_schnell->huge_hidden_states2);
    bm_free_device(flux_schnell->handles[0], flux_schnell->temb);
    bm_free_device(flux_schnell->handles[1], flux_schnell->temb1);
    bm_free_device(flux_schnell->handles[2], flux_schnell->temb2);
    bm_free_device(flux_schnell->handles[0], flux_schnell->image_rotary_emb);
    bm_free_device(flux_schnell->handles[1], flux_schnell->image_rotary_emb1);
    bm_free_device(flux_schnell->handles[2], flux_schnell->image_rotary_emb2);
    bm_free_device(flux_schnell->handles[2], flux_schnell->predict_noise);
    bmrt_destroy(flux_schnell->models[0]);
    bmrt_destroy(flux_schnell->models[1]);
    bmrt_destroy(flux_schnell->models[2]);
    free(flux_schnell);
    flux_schnell = NULL;
    return 0;
}

// struct flux_schnell_with_multi_device * flux_schnell_with_multi_device_init(const char** filename, int device_num, int* device_ids);
// // inline int flux_schnell_head(struct flux_schnell *flux_schnell);
// // inline int flux_schnell_block(struct flux_schnell *flux_schnell, int block_idx);
// // inline int flux_schnell_tail(struct flux_schnell *flux_schnell);

// inline int flux_schnell_with_multi_device_head(struct flux_schnell_with_multi_device *flux_schnell);
// inline int flux_schnell_with_multi_device_block(struct flux_schnell_with_multi_device *flux_schnell, int block_idx);
// inline int flux_schnell_with_multi_device_tail(struct flux_schnell_with_multi_device *flux_schnell);

struct flux_schnell * flux_schnell_init(const char* filename, int device_id)
{
    struct flux_schnell *flux_schnell = (struct flux_schnell *)calloc(1, sizeof(struct flux_schnell));
    flux_schnell->handle                 = get_handle(device_id);
    flux_schnell->device_id              = device_id;
    flux_schnell->model                  = bmrt_create(flux_schnell->handle);
    SD3_ASSERT(bmrt_load_bmodel(flux_schnell->model, filename)==true);
    // auto num_nets              = bmrt_get_network_number(flux_schnell->model);
    flux_schnell->net_head               = bmrt_get_network_info(flux_schnell->model, FLUX_SCHNELL_PREFIX"head");
    flux_schnell->net_tail               = bmrt_get_network_info(flux_schnell->model, FLUX_SCHNELL_PREFIX"tail");
    flux_schnell->net_transform_block    = bmrt_get_network_info(flux_schnell->model, FLUX_SCHNELL_PREFIX"transformer_block_0");
    flux_schnell->net_simple_block       = bmrt_get_network_info(flux_schnell->model, FLUX_SCHNELL_PREFIX"single_transformer_block_0");
    
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->init_states,                 flux_schnell->net_head->max_input_bytes[0] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->timestep,                    flux_schnell->net_head->max_input_bytes[1] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->guidance,                    flux_schnell->net_head->max_input_bytes[1] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->pooled_projections,          flux_schnell->net_head->max_input_bytes[2] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->init_encoder_hidden_states,  flux_schnell->net_head->max_input_bytes[3] ) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->huge_hidden_states,          flux_schnell->net_simple_block->max_input_bytes[0] ) == 0 );
    
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->temb,                        flux_schnell->net_transform_block->max_input_bytes[2]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->image_rotary_emb,            flux_schnell->net_transform_block->max_input_bytes[3]) == 0 );

    bm_device_mem_offset(&flux_schnell->huge_hidden_states, 
                          &flux_schnell->encoder_hidden_states,
                          0, 
                          flux_schnell->net_transform_block->max_output_bytes[0]);

    bm_device_mem_offset(&flux_schnell->huge_hidden_states, 
                          &flux_schnell->hidden_states, 
                          flux_schnell->net_transform_block->max_output_bytes[0],
                          flux_schnell->net_transform_block->max_output_bytes[1]);
    SD3_ASSERT( bm_malloc_device_byte(flux_schnell->handle, &flux_schnell->predict_noise, flux_schnell->net_tail->max_output_bytes[0]) == 0 );
    flux_schnell->buffer = (void*)malloc(flux_schnell->net_tail->max_output_bytes[0]);
    return flux_schnell;
}

int flux_schnell_run(struct flux_schnell *flux_schnell, void* input0, void* input1, void* input2, void* input3, void* rotary_emb, void* output, int do_unpack)
{
    FUNC_TIME_START;
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(flux_schnell->net_head->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(flux_schnell->net_head->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   flux_schnell->init_states,                 flux_schnell->net_head->input_dtypes[0],   flux_schnell->net_head->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, flux_schnell->timestep,                    flux_schnell->net_head->input_dtypes[1],   flux_schnell->net_head->stages[0].input_shapes[1]);
    // bmrt_tensor_with_device(input_tensors+2, flux_schnell->guidance,                    flux_schnell->net_head->input_dtypes[2],   flux_schnell->net_head->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(input_tensors+2, flux_schnell->pooled_projections,          flux_schnell->net_head->input_dtypes[2],   flux_schnell->net_head->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(input_tensors+3, flux_schnell->init_encoder_hidden_states,  flux_schnell->net_head->input_dtypes[3],   flux_schnell->net_head->stages[0].input_shapes[3]);

    // s2d
    SD3_ASSERT(bm_memcpy_s2d(flux_schnell->handle, flux_schnell->init_states, input0) == 0);
    SD3_ASSERT(bm_memcpy_s2d(flux_schnell->handle, flux_schnell->timestep,    input1) == 0);
    // SD3_ASSERT(bm_memcpy_s2d(flux_schnell->handle, flux_schnell->guidance, input2) == 0);
    SD3_ASSERT(bm_memcpy_s2d(flux_schnell->handle, flux_schnell->pooled_projections, input2) == 0);
    SD3_ASSERT(bm_memcpy_s2d(flux_schnell->handle, flux_schnell->init_encoder_hidden_states, input3) == 0);

    bmrt_tensor_with_device(output_tensors,  flux_schnell->temb,                        flux_schnell->net_head->output_dtypes[0],  flux_schnell->net_head->stages[0].output_shapes[0]);
    bmrt_tensor_with_device(output_tensors+1,flux_schnell->encoder_hidden_states,        flux_schnell->net_head->output_dtypes[1],  flux_schnell->net_head->stages[0].output_shapes[1]);
    bmrt_tensor_with_device(output_tensors+2,flux_schnell->hidden_states,               flux_schnell->net_head->output_dtypes[2],  flux_schnell->net_head->stages[0].output_shapes[2]);
    {
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_schnell->model, flux_schnell->net_head->name, input_tensors, flux_schnell->net_head->input_num, output_tensors, flux_schnell->net_head->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_schnell->handle);
        DOMAIN_TIME_END("flux_schnell_head");
    }
        // input: hidden_states, encoder_hidden_states, temb, image_rotary_emb
        // output: encoder_hidden_states, hidden_states
        input_tensors  = (bm_tensor_t*) realloc(input_tensors, flux_schnell->net_transform_block->input_num * sizeof(bm_tensor_t));
        output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_schnell->net_transform_block->output_num * sizeof(bm_tensor_t));
        bmrt_tensor_with_device(input_tensors,   flux_schnell->hidden_states,               flux_schnell->net_transform_block->input_dtypes[0], flux_schnell->net_transform_block->stages[0].input_shapes[0]);
        bmrt_tensor_with_device(input_tensors+1, flux_schnell->encoder_hidden_states,        flux_schnell->net_transform_block->input_dtypes[1], flux_schnell->net_transform_block->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(input_tensors+2, flux_schnell->temb,                        flux_schnell->net_transform_block->input_dtypes[2], flux_schnell->net_transform_block->stages[0].input_shapes[2]);
        bmrt_tensor_with_device(input_tensors+3, flux_schnell->image_rotary_emb,            flux_schnell->net_transform_block->input_dtypes[3], flux_schnell->net_transform_block->stages[0].input_shapes[3]);
        // s2d image_rotary_emb
        SD3_ASSERT(bm_memcpy_s2d(flux_schnell->handle, flux_schnell->image_rotary_emb, rotary_emb) == 0);

        bmrt_tensor_with_device(output_tensors,  flux_schnell->encoder_hidden_states,        flux_schnell->net_transform_block->output_dtypes[0], flux_schnell->net_transform_block->stages[0].output_shapes[0]);
        bmrt_tensor_with_device(output_tensors+1,flux_schnell->hidden_states,               flux_schnell->net_transform_block->output_dtypes[1], flux_schnell->net_transform_block->stages[0].output_shapes[1]);
        char name[100];
    // transform block
    loop(i, FLUX_SCHNELL_NUM_BLOCKS)
    {
        // name: schnell_transformer_block_0
        sprintf(name, FLUX_SCHNELL_PREFIX"transformer_block_%d", i);
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_schnell->model, name, input_tensors, flux_schnell->net_transform_block->input_num, output_tensors, flux_schnell->net_transform_block->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_schnell->handle);
        DOMAIN_TIME_END(name);
        strcpy(name, "");
    }
    // simple transform block
    // input : huge_hidden_states, temb, image_rotary_emb
    // output: huge_hidden_states
    // name  : schnell_single_transformer_block_0
    input_tensors  = (bm_tensor_t*) realloc(input_tensors, flux_schnell->net_simple_block->input_num * sizeof(bm_tensor_t));
    output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_schnell->net_simple_block->output_num * sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   flux_schnell->huge_hidden_states,          flux_schnell->net_simple_block->input_dtypes[0], flux_schnell->net_simple_block->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, flux_schnell->temb,                        flux_schnell->net_simple_block->input_dtypes[1], flux_schnell->net_simple_block->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(input_tensors+2, flux_schnell->image_rotary_emb,            flux_schnell->net_simple_block->input_dtypes[2], flux_schnell->net_simple_block->stages[0].input_shapes[2]);
    bmrt_tensor_with_device(output_tensors,  flux_schnell->huge_hidden_states,          flux_schnell->net_simple_block->output_dtypes[0], flux_schnell->net_simple_block->stages[0].output_shapes[0]);
    
    loop(i, FLUX_SCHNELL_SIMPLE_NUM_BLOCKS)
    {
        sprintf(name, FLUX_SCHNELL_PREFIX"single_transformer_block_%d", i);
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_schnell->model, name, input_tensors, flux_schnell->net_simple_block->input_num, output_tensors, flux_schnell->net_simple_block->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_schnell->handle);
        DOMAIN_TIME_END(name);
        strcpy(name, "");
    }
    // tail
    // input : hidden_states, temb
    // output: hidden_states
    // name  : schnell_tail
    input_tensors = (bm_tensor_t*) realloc(input_tensors, flux_schnell->net_tail->input_num * sizeof(bm_tensor_t));
    output_tensors = (bm_tensor_t*) realloc(output_tensors, flux_schnell->net_tail->output_num * sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,   flux_schnell->hidden_states,               flux_schnell->net_tail->input_dtypes[0], flux_schnell->net_tail->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(input_tensors+1, flux_schnell->temb,                        flux_schnell->net_tail->input_dtypes[1], flux_schnell->net_tail->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(output_tensors,  flux_schnell->predict_noise,               flux_schnell->net_tail->output_dtypes[0], flux_schnell->net_tail->stages[0].output_shapes[0]);
    {
        DOMAIN_TIME_START;
        auto ret = bmrt_launch_tensor_ex(flux_schnell->model, flux_schnell->net_tail->name, input_tensors, flux_schnell->net_tail->input_num, output_tensors, flux_schnell->net_tail->output_num, true, false);
        SD3_ASSERT(ret);
        bm_thread_sync(flux_schnell->handle);
        DOMAIN_TIME_END("flux_schnell_tail");
    }
    free(input_tensors);
    free(output_tensors);
    // d2s
    // printf("do_unpack %d\n", do_unpack);
    if(do_unpack == 1)
    {
        SD3_ASSERT(bm_memcpy_d2s(flux_schnell->handle, flux_schnell->buffer, flux_schnell->predict_noise) == 0);
        convert_pathify_to_latent(flux_schnell->buffer, output, 1, 64, 64, 2, 2, 16, 4);
    }else{
        // printf("do not unpack\n");
        SD3_ASSERT(bm_memcpy_d2s(flux_schnell->handle, output, flux_schnell->predict_noise) == 0);
    }
    FUNC_TIME_END;
    return 0;
}

int flux_schnell_free(struct flux_schnell *flux_schnell)
{
    bm_free_device(flux_schnell->handle, flux_schnell->init_states);
    bm_free_device(flux_schnell->handle, flux_schnell->timestep);
    bm_free_device(flux_schnell->handle, flux_schnell->guidance);
    bm_free_device(flux_schnell->handle, flux_schnell->pooled_projections);
    bm_free_device(flux_schnell->handle, flux_schnell->init_encoder_hidden_states);
    bm_free_device(flux_schnell->handle, flux_schnell->huge_hidden_states);
    bm_free_device(flux_schnell->handle, flux_schnell->temb);
    bm_free_device(flux_schnell->handle, flux_schnell->predict_noise);
    bmrt_destroy(flux_schnell->model);
    // bm_dev_free(flux_schnell->handle);
    free(flux_schnell);
    flux_schnell = NULL;
    return 0;
}

#ifdef __cplusplus
}
#endif
