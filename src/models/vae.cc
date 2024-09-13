
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vae_decoder {
    int device_id;
    void* model;
    bm_device_mem_t latent;
    bm_device_mem_t img;
    bm_handle_t handle;
    const bm_net_info_t* net;
};

struct vae_decoder* vae_decoder_init(const char* filename, int device_id);
int vae_decoder_run(struct vae_decoder *decoder, void* latent, void* img, bool do_post_process);
int vae_decoder_free(struct vae_decoder *decoder);


struct vae_decoder* vae_decoder_init(const char* filename, int device_id){
    struct vae_decoder *decoder = (struct vae_decoder *)calloc(1, sizeof(struct vae_decoder));
    decoder->device_id = device_id;
    decoder->handle = get_handle(device_id);
    decoder->model = bmrt_create(decoder->handle);
    SD3_ASSERT(bmrt_load_bmodel(decoder->model, filename)==true);
    auto num_nets = bmrt_get_network_number(decoder->model);
    decoder->net = bmrt_get_network_info(decoder->model, "vae_decoder");
    SD3_ASSERT( bm_malloc_device_byte(decoder->handle, &decoder->latent, decoder->net->max_input_bytes[0]) == 0 );
    SD3_ASSERT( bm_malloc_device_byte(decoder->handle, &decoder->img, decoder->net->max_output_bytes[0]) == 0 );
    return decoder;
}

int vae_decoder_run(struct vae_decoder *decoder, void* latent, void* img, bool do_post_process){
    FUNC_TIME_START;
    SD3_ASSERT( bm_memcpy_s2d(decoder->handle, decoder->latent, latent) == 0 );
    bm_tensor_t* input_tensors  = (bm_tensor_t*) calloc(decoder->net->input_num, sizeof(bm_tensor_t));
    bm_tensor_t* output_tensors = (bm_tensor_t*) calloc(decoder->net->output_num, sizeof(bm_tensor_t));
    bmrt_tensor_with_device(input_tensors,  decoder->latent, decoder->net->input_dtypes[0], decoder->net->stages[0].input_shapes[0]);
    bmrt_tensor_with_device(output_tensors, decoder->img, decoder->net->output_dtypes[0], decoder->net->stages[0].output_shapes[0]);
    auto ret = bmrt_launch_tensor_ex(decoder->model, decoder->net->name, input_tensors, decoder->net->input_num, output_tensors, decoder->net->output_num, true, false);
    SD3_ASSERT(ret);
    bm_thread_sync(decoder->handle);
    SD3_ASSERT( bm_memcpy_d2s(decoder->handle, img, decoder->img) == 0 );
    if(do_post_process){
        SD3_UNUSED(do_post_process);
    }
    free(input_tensors);
    free(output_tensors);
    FUNC_TIME_END;
    return 0;
}

int vae_decoder_free(struct vae_decoder *decoder){
    bm_free_device(decoder->handle, decoder->latent);
    bm_free_device(decoder->handle, decoder->img);
    bmrt_destroy(decoder->model);
    free(decoder);
    decoder = NULL;
    return 0;
}

#ifdef __cplusplus
}
#endif




