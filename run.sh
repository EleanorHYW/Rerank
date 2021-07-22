CUDA_VISIBLE_DEVICES=0 python Train.py --train_size 0.98 --valid_size 0.01 \
--eval_size 0.01 --batch_size 128 --n_epochs 10 --gpu --max_len 500 \
--embedding_size 136 --hiddens 128 --n_lstms 1 --dropout 0.1 --L2 \
--ckpt_file ./models/base_bsz_128_h_128_dp_0.1_L2_1_ckp_0.pt \