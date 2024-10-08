import torch
torch.manual_seed(0)

# ==================================================
# export clip
# ==================================================
from transformers import CLIPTextModel
text_encoder  = CLIPTextModel.from_pretrained('/data/aigc/FLUX.1-schnell/text_encoder/')
max_seq_len = 77
test_input = torch.randint(0,1000,(1,max_seq_len))

from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
causal_attention = _create_4d_causal_attention_mask([1,max_seq_len], torch.float32, device="cpu")
clip0_model = text_encoder
for para in clip0_model.parameters():
    para.requires_grad=False
clip0_model = clip0_model.eval()

class CLIPHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = clip0_model.text_model.embeddings
    def forward(self, x):
        return self.emb(x)
head = CLIPHead()
# torch.onnx.export(head, test_input, "/data/aigc/flux/models/clip/head.onnx",opset_version=16)
torch.jit.trace(head, test_input).save("/data/aigc/flux/models/clip/head.pt")
input1 = head(test_input)

class CLIPBlock(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = clip0_model.text_model.encoder.layers[idx]
    def forward(self, x):
        return self.block(x, None, causal_attention)[0]
for i in range(12):
    block0 = CLIPBlock(i)
    torch.jit.trace(block0, input1).save( f"/data/aigc/flux/models/clip/block_{i}.pt")
    # torch.onnx.export(block0, input1, f"/data/aigc/flux/models/clip/block_{i}.onnx",opset_version=16)
    input1 = block0(input1)

class CLIPTail(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer_norm = clip0_model.text_model.final_layer_norm
        # self.text_proj = clip0_model.text_projection
    def forward(self, x, inputs):
        x = self.final_layer_norm(x)
        x = x[0,inputs.argmax(dim=-1)]
        # x = self.text_proj(x)
        return x
# res_t = clip0_model.text_model.final_layer_norm(input1)
tail = CLIPTail()
torch.jit.trace(tail, (input1, test_input)).save("/data/aigc/flux/models/clip/tail.pt")
# torch.onnx.export(clip0_model.text_model.final_layer_norm, input1, f"/data/aigc/flux/models/clip/tail.onnx",opset_version=16)


# ==================================================
# export vae
# ==================================================
from diffusers import AutoencoderKL,AutoencoderTiny
vae = AutoencoderTiny.from_pretrained("/data/aigc/models/taef1/")
vae= vae.eval()
for para in vae.parameters():
    para.requires_grad = False
class VAE_Decoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.vae = vae
        pass

    def forward(self, hidden_states):
        hidden_states = (hidden_states / vae.config.scaling_factor) + vae.config.shift_factor
        res = self.vae.decode(hidden_states)[0]
        return res

vae_decoder = VAE_Decoder()

fake_inputs = torch.randn(1, 16, 128, 128)
torch.onnx.export(vae_decoder, fake_inputs, "/data/aigc/flux/models/tiny_vae//vae_decoder.onnx")

# ==================================================
# export t5
# ==================================================
from transformers import T5EncoderModel
text_encoder2 = T5EncoderModel.from_pretrained("/data/aigc/FLUX.1-schnell/text_encoder_2/")
t5_model = text_encoder2
t5_encoder = [[   71,  1712,  3609,     3,     9,  1320,    24,   845, 21820,   296,
             1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0]]
t5_encoder[0] = t5_encoder[0] + [0] * (512-77)
t5_encoder_inputs = torch.tensor(t5_encoder,dtype=torch.int32)
t5_model.eval()
for para in t5_model.parameters():
    para.requires_grad = False

class T5Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = t5_model.encoder.embed_tokens
    def forward(self, test_input):
        return self.emb(test_input)
        
t5head = T5Head().eval()
# t5_encoder_inputs = torch.randint(0, 1000, (1, 512))
input1 = t5head(t5_encoder_inputs)
# torch.onnx.export(t5head, t5_encoder_inputs, "/data/aigc/flux/models/t5/head.onnx")
t5model = t5_model
temp_value = t5model.encoder.block[0].layer[0](t5model.encoder.embed_tokens(t5_encoder_inputs))[2:][0].detach().requires_grad_(False)
t = torch.zeros(1,1,1,512)
class T5block0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = t5model.encoder.block[0].layer
        
    def forward(self, test_input):
        output = self.block0[0](test_input, t, temp_value)
        hidden_states = output[0]
        hidden_states = self.block0[-1](hidden_states)
        return hidden_states
    
t5_block_0 = T5block0()
t5block0 = T5block0().eval()
torch.onnx.export(t5block0, input1, "/data/aigc/flux/models/t5/block_0.onnx")

class T5blocknext(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = t5_model.encoder.block[idx].layer
    def forward(self, test_input):
        next_block = self.block
        output = next_block[0](test_input,t, temp_value)
        hidden_states = output[0]
        hidden_states = next_block[1](hidden_states)
        return hidden_states
from tqdm.auto import tqdm
t5_next_blocks = []
for i in tqdm(range(1,23)):
    block = T5blocknext(i).eval().bfloat16()
    # torch.onnx.export(block, hidden_states, f"/data/aigc/demos/sd3models/t5/t5_encoder_block{i}.onnx")
    torch.jit.trace(block, hidden_states).save(f"/data/aigc/flux/models/t5/block{i}.pt")
    hidden_states = block(hidden_states)
    # print(i, hidden_states.max(), hidden_states.min() )

torch.jit.trace(t5model.encoder.final_layer_norm, hidden_states).save("/data/aigc/flux/models/t5/tail.pt")


# ==================================================
# export transformer
# ==================================================
from diffusers import FluxTransformer2DModel
FluxWeightPath = "/data/aigc/FLUX.1-schnell/flux1-schnell.safetensors"
flux = FluxTransformer2DModel.from_single_file(FluxWeightPath)
for para in flux.parameters():
    para.requires_grad = False 
flux = flux.eval()

hidden_states_head = torch.randn(1, 4096, 64)
timestep = torch.tensor([0.999])
guidance = torch.randn(1)                 
pooled_projections = torch.randn(1, 768)
encoder_hidden_states_head = torch.randn(1, 512, 4096)
head_input = (hidden_states_head,timestep,pooled_projections,encoder_hidden_states_head)

temb = torch.randn(1,3072)
hidden_states = torch.randn(1, 4096, 3072)
encoder_hidden_states = torch.randn(1, 512, 3072)
image_rotary_emb = torch.randn(1, 4608,1, 64, 2, 2)
s_hidden_states = torch.randn(1, 4608, 3072)

test_input_single = (s_hidden_states,temb,image_rotary_emb)
test_input = (hidden_states,encoder_hidden_states,temb,image_rotary_emb)
test_input_tail = (hidden_states,temb)
class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.pos_embed = flux.pos_embed
        self.time_text_embed = flux.time_text_embed
        self.context_embedder =flux.context_embedder
        self.x_embedder =flux.x_embedder
        # self.silu = torch.nn.SiLU()
    def forward(
        self,
        hidden_states, # torch.Size([1, 4096, 64])
        timestep, # torch.Size([1])
        # guidance, # torch.Size([1])
        pooled_projections, # torch.Size([1, 768])
        # img_ids, # torch.Size([1, 4096, 3]
        # txt_ids, # torch.Size([1, 512, 3])
        encoder_hidden_states, # torch.Size([1, 512, 4096])
        guidance=None,
    ):
        hidden_states = self.x_embedder(hidden_states)
        # timestep = timestep.to(hidden_states.dtype) * 1000
        # if guidance is not None:
        #     guidance = guidance.to(hidden_states.dtype) * 1000
        # else:
        #     guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        # temb = self.silu(temb)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        # txt_ids = txt_ids.expand(img_ids.size(0), -1, -1)
        # ids = torch.cat((txt_ids, img_ids), dim=1)
        # image_rotary_emb = self.pos_embed(ids)
        return temb, encoder_hidden_states, hidden_states
    
# temb, encoder_hidden_states, hidden_states = head(hidden_states_head, timestep, pooled_projections, encoder_hidden_states_head)
head = Head()
torch.jit.trace(head,head_input).save("/data/aigc/flux/models/transform//trans_head.pt")

class Tail(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_out = flux.norm_out
        self.proj_out = flux.proj_out
    def forward(
        self,
        hidden_states, # torch.Size([1, 4096, 3072])
        temb, # torch.Size([1, 3072])
    ):
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return output
tail = Tail()
torch.jit.trace(tail,test_input_tail).save("/data/aigc/flux/models/transform/trans_tail.pt")

from tqdm.autonotebook import tqdm
class Flux_BLOCK(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = flux.transformer_blocks[idx]

    def forward(self,hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        encoder_hidden_states, hidden_states= self.block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
        return encoder_hidden_states, hidden_states
transformer_blocks = []
for i in tqdm(range(19)):
    transformer_blocks.append(Flux_BLOCK(i).float())
    traced_model = torch.jit.trace(transformer_blocks[-1],test_input)
    traced_model.save(f"/data/aigc/flux/models/transform/transformer_block_{i}.pt")
    # torch.onnx.export(
    #     transformer_blocks[-1],
    #     test_input,
    #     f"/data/aigc/flux/ptWithW/transformer_block_{i}_weight.onnx"
    # )
    # encoder_hidden_states, hidden_states = transformer_blocks[-1](hidden_states, encoder_hidden_states, temb, ids_emb)
    # print(hidden_states.max(), hidden_states.min())
    # print(encoder_hidden_states.max(), hidden_states.min())
    # print(a.shape,b.shape)

class Flux_SingleTransformerBlock(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = flux.single_transformer_blocks[idx]
    def forward(
        self,
        hidden_states,
        temb,
        image_rotary_emb=None,
    ):
        hidden_states = self.block(hidden_states,temb,image_rotary_emb)
        return hidden_states
single_transformer_blocks = []
for i in tqdm(range(38)):
    single_transformer_blocks.append(Flux_SingleTransformerBlock(i).float())
    traced_model = torch.jit.trace(single_transformer_blocks[-1],test_input_single)
    traced_model.save(f"/data/aigc/flux/models/transform/single_transformer_block_{i}.pt")
    # torch.onnx.export(
    #     single_transformer_blocks[-1],
    #     test_input_single,
    #     f"/data/aigc/flux/ptWithW/single_transformer_block_{i}_weight.onnx"
    # )
    # hidden_states = single_transformer_blocks[-1](*test_input_single)