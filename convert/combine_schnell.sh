
# combine t5
# source /etc/profile
# pushd ./clip
# bmodels=`ls ./ | grep ".bmodel$" | grep -v "combined.bmodel"`

# tpu_model --combine $bmodels -o ../combine/clip_combined.bmodel
# popd

# pushd ./t5x256
# bmodels=`ls ./ | grep W4BF16 | grep ".bmodel$" | grep -v "combined.bmodel"`
# head="head_F16.bmodel"
# tail="tail_F16.bmodel"
# tpu_model --combine $bmodels $head $tail -o ../combine/t5x256_combined.bmodel
# popd

# pushd ./transform
# bmodels=`ls ./ | grep W4BF16 | grep ".bmodel$"`
# head="head.bmodel"
# tail="tail.bmodel"
# tpu_model --combine $bmodels $head $tail -o ../combine/schnell_w4bf16_transform_combined.bmodel
# popd


# pushd ./transform
# name=""
# for i in $(seq 0 12);
# do
#     name=$name"transformer_block_$i""_BF16.bmodel "
# done
# head="head.bmodel"
# name=$name" "$head
# echo $name
# tpu_model --combine $name -o ../combine/schnell_bf16_transform_combined_device0.bmodel
# popd

# pushd ./transform
# name=""
# for i in $(seq 13 18);
# do
#     name=$name"transformer_block_$i""_BF16.bmodel "
# done
# for i in $(seq 0 27);
# do
#     name=$name"single_transformer_block_$i""_BF16.bmodel "
# done
# echo $name
# tpu_model --combine $name -o ../combine/schnell_bf16_transform_combined_device1.bmodel
# popd


# pushd ./transform
# name=""
# for i in $(seq 28 37);
# do
#     name=$name"single_transformer_block_$i""_BF16.bmodel "
# done
# tail="tail.bmodel"
# echo $name
# name=$name" "$tail
# tpu_model --combine $name -o ../combine/schnell_bf16_transform_combined_device2.bmodel
# popd

pushd ./trans_256_768
name=""
for i in $(seq 0 37);
do
    name=$name"single_transformer_block_$i""_W4BF16.bmodel "
done
for i in $(seq 0 18);
do
    name=$name"transformer_block_$i""_W4BF16.bmodel "
done
head="head_BF16.bmodel"
tail="tail_BF16.bmodel"
name=$name" "$tail
name=$name" "$head
echo $name
tpu_model --combine $name -o ../combine/schnell_w4bf16_256x768_transform_combined.bmodel
popd