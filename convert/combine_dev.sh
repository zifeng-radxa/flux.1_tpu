


# pushd ./dev_transformer/
# bmodels=`ls ./ | grep W4BF16 | grep ".bmodel$"`
# head="head_F16.bmodel"
# tail="tail_BF16.bmodel"
# tpu_model --combine $bmodels $head $tail -o ../combine/dev_transform_combined.bmodel
# popd

pushd ./dev_transformer/
# name=""
# for i in $(seq 0 12);
# do
#     name=$name"trans_block_$i""_BF16.bmodel "
# done
# head="head_F16.bmodel"
# name=$name" "$head
# echo $name
# tpu_model --combine $name -o ../combine/dev_transform_combined_device0.bmodel
# popd

# name=""
# for i in $(seq 13 18);
# do
#     name=$name"trans_block_$i""_BF16.bmodel "
# done
# for i in $(seq 0 27);
# do
#     name=$name"single_trans_block_$i""_BF16.bmodel "
# done
# echo $name
# tpu_model --combine $name -o ../combine/dev_transform_combined_device1.bmodel
# popd

# name=""
# for i in $(seq 28 37);
# do
#     name=$name"single_trans_block_$i""_BF16.bmodel "
# done
# tail="tail_BF16.bmodel"
# echo $name
# name=$name" "$tail
# tpu_model --combine $name -o ../combine/dev_transform_combined_device2.bmodel
name=""
for i in $(seq 0 37);
do
    name=$name"single_trans_block_$i""_W4F16.bmodel "
done
for i in $(seq 0 18);
do
    name=$name"trans_block_$i""_W4F16.bmodel "
done
head="head_F16.bmodel"
tail="tail_F16.bmodel"
name=$name" "$tail
name=$name" "$head
echo $name
tpu_model --combine $name -o ../combine/dev_w4f16_transform_combined.bmodel

popd




