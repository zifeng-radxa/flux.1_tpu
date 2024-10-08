set -ex


pushd ./vae
name=vae_decoder
shape=[[1,16,128,128]]
quant="F16"
model_transform.py --model_name $name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
quant="BF16"
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`
popd


# clip
pushd ./clip
name=head
shape=[[1,77]]
quant="F16"
prefix=clip
model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def clip_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $prefix'_'$name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=tail
shape=[[77,768]]
quant="F16"
prefix=clip
model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def clip_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $prefix'_'$name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

block_num=11
for i in $(seq 0 $block_num);
do
    name=block_$i
    shape=[[1,77,768]]
    quant="F16"
    model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def clip_$name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $prefix'_'$name'_'$quant.bmodel
    rm -r `ls | grep -E 'npz|json|net_0\.profile'`
done
popd

# # transform dev
pushd ./dev_transformer

name=head
shape=[[1,4096,64],[1],[1],[1,768],[1,512,4096]]
quant="BF16"
model_transform.py --model_name dev_$name --input_shape $shape --model_def trans_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name"_"$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=tail
shape=[[1,4096,3072],[1,3072]]
quant="BF16"
model_transform.py --model_name dev_$name --input_shape $shape --model_def trans_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name"_"$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=trans_block_
block_num=18
for i in $(seq 0 $block_num);
do
    # block_id=$i
    name=trans_block_$i
    shape=[[1,4096,3072],[1,512,3072],[1,3072],[1,4608,1,64,2,2]]
    quant="W4BF16"
    model_transform.py --model_name dev_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="W8BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    rm -r `ls | grep -E 'npz|json|net_0\.profile'`
done

name=single_trans_block_
block_num=37
for i in $(seq 0 $block_num);
do
    name=single_trans_block_$i
    shape=[[1,4608,3072],[1,3072],[1,4608,1,64,2,2]]
    quant="W4BF16"
    model_transform.py --model_name dev_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="W8BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    rm -r `ls | grep -E 'npz|json|net_0\.profile'`
done
popd

pushd ./t5

name=head
shape=[[1,512]]
# shape=[[1,256]]
quant="F16"
model_transform.py --model_name t5_$name --input_shape $shape --model_def t5_encoder_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=tail
shape=[[1,512,4096]]
quant="F16"
model_transform.py --model_name t5_$name --input_shape $shape --model_def t5_encoder_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=block
block_num=23
quant="W4BF16"
for i in $(seq 0 0);
do
    name=block_$i
    shape=[[1,512,4096]]
    # shape=[[1,256,4096]]
    model_transform.py --model_name t5_$name --input_shape $shape --model_def t5_encoder_$name.onnx --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="W8BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="F16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    rm -r `ls | grep -E 'npz|json|net_0\.profile'`
done

name=block
block_num=23
quant="W4BF16"
for i in $(seq 1 $block_num);
do
    name=block_$i
    shape=[[1,512,4096]]
    # shape=[[1,256,4096]]
    model_transform.py --model_name t5_$name --input_shape $shape --model_def t5_encoder_$name.onnx --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="W8BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="F16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    rm -r `ls | grep -E 'npz|json|net_0\.profile'`
done

popd