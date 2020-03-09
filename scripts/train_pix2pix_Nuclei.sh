set -ex

python train.py --dataroot /share_hd1/db/Nuclei --name brats_gan_nuclei_withoutL1 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 0 --delta_perceptual 10 --dataset_mode nuclei --pool_size 0 --gpu_ids 2 --batch_size 12 --num_threads 0 --continue_train --niter=200 --niter_decay=200



