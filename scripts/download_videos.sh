#!/usr/bin/env bash
wget -O videos.tgz "www.vision.ee.ethz.ch/datasets_extra/ewap_dataset_full.tgz"
mkdir temp
tar -C temp -zxf videos.tgz
mv temp/ewap_dataset/seq_eth/seq_eth.avi scenes_and_matrices/eth.avi
mv temp/ewap_dataset/seq_hotel/seq_hotel.avi scenes_and_matrices/hotel.avi
rm -rf temp
rm -f videos.tgz
echo "Done, videos are in scenes_and_matrices folder."