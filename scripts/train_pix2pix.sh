set -ex

python train.py --dataroot /share_hd1/db/BRATS/2018/tumor_size_split_10 --name brats_asyndgan_verticle_flip --model asyndgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_split --pool_size 0 --gpu_ids 1 --batch_size 1 --num_threads 0




