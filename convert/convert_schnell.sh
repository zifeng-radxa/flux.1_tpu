set -ex

# pushd ./tiny_vae_768
pushd ./tiny_vae
name=vae_decoder
shape=[[1,16,128,128]] # for 1024*1024
# shape=[[1,16,96,96]] # for 768*768
quant="F16"
model_transform.py --model_name $name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
quant="BF16"
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`


clip
pushd ./clip
name=head
shape=[[1,77]]
quant="F16"
prefix=clip
model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $prefix'_'$name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

block_num=11
for i in $(seq 0 $block_num);
do
    name=block_$i
    shape=[[1,77,768]]
    quant="F16"
    model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $prefix'_'$name'_'$quant.bmodel
    rm -r `ls | grep -E 'npz|json|net_0\.profile'`
done

name=tail
shape=[[1,77,768],[1,77]]
quant="F16"
model_transform.py --model_name $prefix'_'$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $prefix'_'$name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`


# transform schnell
pushd ./transform

name=head
shape=[[1,4096,64],[1],[1,768],[1,512,4096]]
quant="BF16"
model_transform.py --model_name schnell_$name --input_shape $shape --model_def trans_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name"_"$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=tail
shape=[[1,4096,3072],[1,3072]]
quant="BF16"
model_transform.py --model_name schnell_$name --input_shape $shape --model_def trans_$name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name"_"$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=transformer_block_
block_num=18
for i in $(seq 0 $block_num);
do
    # block_id=$i
    name=transformer_block_$i
    shape=[[1,4096,3072],[1,512,3072],[1,3072],[1,4608,1,64,2,2]]
    quant="W8BF16"
    model_transform.py --model_name schnell_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="W4BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    rm -r `ls | grep -E 'npz|json|net_0\.profile'`
done

name=single_transformer_block_
block_num=37
for i in $(seq 0 $block_num);
do
    name=single_transformer_block_$i
    shape=[[1,4608,3072],[1,3072],[1,4608,1,64,2,2]]
    quant="W4BF16"
    model_transform.py --model_name schnell_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
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
quant="F16"
model_transform.py --model_name t5_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=tail
shape=[[1,512,4096]]
quant="F16"
model_transform.py --model_name t5_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
rm -r `ls | grep -E 'npz|json|net_0\.profile'`

name=block_
block_num=23
quant="W4BF16"
for i in $(seq 0 0);
do
    name=block_$i
    shape=[[1,512,4096]]
    model_transform.py --model_name t5_$name --input_shape $shape --model_def $name.onnx --mlir $name.mlir
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="W8BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="BF16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    quant="F16"
    model_deploy.py --mlir $name.mlir --quantize $quant --chip bm1684x --model $name'_'$quant.bmodel
    rm -r `ls | grep -E 'npz|json|net_0\.profile'`
done

name=block_
block_num=23
quant="W4BF16"
for i in $(seq 1 $block_num);
do
    name=block_$i
    shape=[[1,512,4096]]
    model_transform.py --model_name t5_$name --input_shape $shape --model_def $name.pt --mlir $name.mlir
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
 