#!/bin/bash

REMOTE_DIR="https://github.com/zifeng-radxa/flux.1_tpu/releases/download/models"
LOCAL_DIR="./"
MODEL="models.tar.gz"

for SUFFIX in aa ab ac ad ae af ag ah ai aj; do
  FILE_NAME="$MODEL.$SUFFIX"
  FILE_URL="$REMOTE_DIR/$FILE_NAME"


  wget "$FILE_URL"
  echo "success download $FILE_NMAE"
done

echo "merge!!"
# 合并文件块
cat "$LOCAL_DIR"/$MODEL.* > $MODEL

# 检查合并文件的 MD5
EXPECTED_MD5="08a7279ce6b361540f12397e453bd33f"
ACTUAL_MD5=$(md5sum $MODEL | awk '{print $1}')

if [ "$EXPECTED_MD5" == "$ACTUAL_MD5" ]; then
  echo "MD5 verification successful."
else
  echo "MD5 verification failed."
  echo "Please check your network and download again"
fi

rm $MODEL.*