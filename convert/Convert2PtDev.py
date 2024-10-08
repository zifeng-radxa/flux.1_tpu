from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel
import torch
torch.manual_seed(0)

pipe = FluxPipeline.from_pretrained("/data/aigc/FLUX.1-dev")

# ==================================================
# export clip
# ==================================================

# from transformers import CLIPTextModel
# text_encoder  = CLIPTextModel.from_pretrained('/data/aigc/FLUX.1-schnell/text_encoder/')

clip = pipe.text_encoder
clip.eval()
for para in clip.parameters():
    para.requires_grad = False

max_seq_len = 77
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
causal_attention = _create_4d_causal_attention_mask([1,max_seq_len], torch.float32, device="cpu")
input = torch.randint(0,1000,(1,77))

class CLIPHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = clip.text_model.embeddings
    def forward(self, x):
        return self.emb(x)
cliphead = CLIPHead()
torch.jit.trace(cliphead, input).save("/data/aigc/flux/ptFiles/clip/clip_head.pt")

input1 = cliphead(input)
class CLIPBlock(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = clip.text_model.encoder.layers[idx]
    def forward(self, x):
        return self.block(x, None, causal_attention, False)[0]
    
for i in range(12):
    block0 = CLIPBlock(i)
    torch.jit.trace(block0, input1).save(f"/data/aigc/flux/ptFiles/clip/clip_block_{i}.pt")
    input1 = block0(input1)
    # print(i,input1.max(),input1.min())

class CLIPTail(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer_norm = clip.text_model.final_layer_norm

    def forward(self, x):
        x = self.final_layer_norm(x)
        return x
input1 = input1[0]
tail = CLIPTail()
torch.jit.trace(tail, input1).save(f"/data/aigc/flux/ptFiles/clip/clip_tail.pt")

# ==================================================
# export vae
# ==================================================
vae = pipe.vae
vae = vae.eval()
for para in vae.parameters():
    para.requires_grad = False
class VAE_Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = vae
        pass

    def forward(self, hidden_states):
        hidden_states = (hidden_states / 0.3611) + 0.1159
        res = self.vae.decode(hidden_states)[0]
        return res

vae_decoder = VAE_Decoder()
fake_inputs = torch.randn(1, 16, 128, 128)
torch.jit.trace(vae_decoder, fake_inputs).save("/data/aigc/flux/ptFiles/vae/vae_decoder.pt")
torch.onnx.export(vae_decoder, fake_inputs, "/data/aigc/flux/ptFiles/vae/vae_decoder.onnx")

# ==================================================
# export t5
# ==================================================
t5_model = pipe.text_encoder_2
t5_model.eval()
for para in t5_model.parameters():
    para.requires_grad = False

import torch
t5_encoder = [[   71,  1712,  3609,     3,     9,  1320,    24,   845, 21820,   296,
             1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0]]
t5_encoder[0] = t5_encoder[0] + [0] * (512-77) # 512=seq len, change it to 256 when needed
t5_encoder_inputs = torch.tensor(t5_encoder,dtype=torch.int32)

class T5Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = t5_model.encoder.embed_tokens
    def forward(self, test_input):
        return self.emb(test_input)
        
t5head = T5Head().eval()
input1 = t5head(t5_encoder_inputs)
torch.jit.trace(t5head, t5_encoder_inputs).save(f"/data/aigc/flux/ptFiles/t5/t5_encoder_head.pt")

temp_value = t5_model.encoder.block[0].layer[0](t5_model.encoder.embed_tokens(t5_encoder_inputs))[2:][0].detach().requires_grad_(False)
t = torch.zeros(1,1,1,512)
class T5block0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = t5_model.encoder.block[0].layer
        
    def forward(self, test_input):
        output = self.block0[0](test_input, t, temp_value) # temp_value
        hidden_states = output[0]
        hidden_states = self.block0[-1](hidden_states)
        return hidden_states
    
t5_block_0 = T5block0()
t5block0 = T5block0().eval()
torch.jit.trace(t5_block_0, input1).save(f"/data/aigc/flux/ptFiles/t5/t5_encoder_block_{0}.pt")
torch.onnx.export(t5_block_0, input1, f"/data/aigc/flux/ptFiles/t5/t5_encoder_block_{0}.onnx")

class T5blocknext(torch.nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.block = t5_model.encoder.block[idx].layer
    def forward(self, test_input):
        next_block = self.block
        output = next_block[0](test_input,t,temp_value)
        hidden_states = output[0]
        hidden_states = next_block[1](hidden_states)
        return hidden_states
    
hidden_states = t5block0(input1)
from tqdm.auto import tqdm
t5_next_blocks = []
for i in tqdm(range(1,24)):
    block = T5blocknext(i).eval().bfloat16()
    torch.onnx.export(block, hidden_states, f"/data/aigc/flux/ptFiles/t5/t5_encoder_block_{i}.onnx")
    # torch.jit.trace(block, hidden_states).save(f"/data/aigc/flux/ptFiles/t5/t5_encoder_block{i}.pt")
    hidden_states = block(hidden_states)
    # print(i, hidden_states.max(), hidden_states.min() )

torch.jit.trace(t5_model.encoder.final_layer_norm, hidden_states).save(f"/data/aigc/flux/ptFiles/t5/t5_encoder_tail.pt")

# ==================================================
# export transformer
# ==================================================
flux = pipe.transformer
for para in flux.parameters():
    para.requires_grad = False

flux = flux.eval()

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
        hidden_states, 
        timestep, 
        guidance, 
        pooled_projections, 
        encoder_hidden_states, 
    ):
        hidden_states = self.x_embedder(hidden_states)
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        # temb = self.silu(temb)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        return temb, encoder_hidden_states, hidden_states
    
head = Head()
hidden_states_head = torch.randn(1, 4096, 64)
timestep = torch.randn(1)
guidance = torch.randn(1)                 
pooled_projections = torch.randn(1, 768)
encoder_hidden_states_head = torch.randn(1, 512, 4096)
head_input = (hidden_states_head,timestep,guidance,pooled_projections,encoder_hidden_states_head)

torch.jit.trace(head,head_input).save(f"/data/aigc/flux/ptFiles/dev_transformer/head.pt")

# temb, encoder_hidden_states, hidden_states = head(hidden_states_head, timestep, guidance, pooled_projections, encoder_hidden_states_head)
# print(temb.max(),temb.min())
# print(encoder_hidden_states.max(),encoder_hidden_states.min())
# print(hidden_states.max(),hidden_states.min())

class Tail(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_out = flux.norm_out
        self.proj_out = flux.proj_out
    def forward(
        self,
        hidden_states,
        temb, 
    ):
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return output
    
tail =Tail()
hidden_states_tail = torch.randn(1, 4096, 3072)
temb = torch.randn(1, 3072)
tail_input = (hidden_states_tail,temb)
torch.jit.trace(tail,tail_input).save(f"/data/aigc/flux/ptFiles/dev_transformer/tail.pt")
# output = tail(hidden_states_tail, temb)
# print(output.max(),output.min())

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
    
temb = torch.randn(1,3072)
hidden_states = torch.randn(1, 4096, 3072)
encoder_hidden_states = torch.randn(1, 512, 3072)
image_rotary_emb = torch.randn(1, 4608, 1, 64, 2, 2)
s_hidden_states = torch.randn(1, 4608, 3072)
test_input_single = (s_hidden_states,temb,image_rotary_emb)
test_input = (hidden_states,encoder_hidden_states,temb,image_rotary_emb)

single_transformer_blocks = []
for i in range(38):
    single_transformer_blocks.append(Flux_SingleTransformerBlock(i).float())
    traced_model = torch.jit.trace(single_transformer_blocks[-1],test_input_single)
    traced_model.save(f"/data/aigc/flux/ptFiles/dev_transformer/single_trans_block_{i}.pt")
    hidden_states = single_transformer_blocks[-1](*test_input_single)
    # print(hidden_states.max(),hidden_states.min())

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
for i in range(19):
    transformer_blocks.append(Flux_BLOCK(i).float())
    traced_model = torch.jit.trace(transformer_blocks[-1],test_input)
    traced_model.save(f"/data/aigc/flux/ptFiles/dev_transformer/trans_block_{i}.pt")
    encoder_hidden_states, hidden_states = transformer_blocks[-1](*test_input)
    # print(hidden_states.max(),hidden_states.min())
    # print(encoder_hidden_states.max(),encoder_hidden_states.min())