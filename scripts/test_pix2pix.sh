set -ex


python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/for_seg --name nuclei_asyndgan_withoutL1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 300 --results_dir results/nuclei_asyndgan_withoutL1



