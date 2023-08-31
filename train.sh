# python eval_dinov2_setr_inov_vits.py \
#         --data_path /nfs/home/mwei/mmsegmentation/data/robo \
#         --output_dir /nfs/home/mwei/SelfSL4MIS_experiment/eval_dinov2_inov_4_last_addition_in_fm_with_fusion_3_layers_vits \
#         --arch vit_base \
#         --patch_size 14 \
#         --n_last_blocks 4 \
#         --imsize 588 \
#         --lr 0.01 \
#         --config_file dinov2/configs/eval/vits14_pretrain.yaml \
#         --pretrained_weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
#         --num_workers 2 \
#         --epochs 500 \

# python eval_dinov2_mla.py \
#         --data_path /nfs/home/mwei/mmsegmentation/data/robo \
#         --output_dir /nfs/home/mwei/SelfSL4MIS_experiment/eval_dinov2_mla_head_loss_dcce \
#         --arch vit_base \
#         --patch_size 14 \
#         --n_last_blocks 4 \
#         --imsize 588 \
#         --lr 0.01 \
#         --config_file dinov2/configs/eval/vits14_pretrain.yaml \
#         --pretrained_weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
#         --num_workers 2 \
#         --epochs 500 \
ROBOMIS_DIR=.../data/robo python src/train_mla.py
