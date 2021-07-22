CUDA_VISIBLE_DEVICES=0 python Train.py --train_size 0.98 --valid_size 0.01 \
--eval_size 0.01 --batch_size 128 --n_epochs 10 --gpu --max_len 500 \
--embedding_size 136 --hiddens 128 --n_lstms 1 \
--ckpt_file ./models/checkpoint_base_0.pt

#python infer.py --batch_size 8 --embedding_size 4 --hiddens 32 --n_lstms 1 --n_test 100 --ckpt_file ./models/checkpoint_0.pkl --max_len 10 --history_len 5