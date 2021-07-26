
echo "start training seq2slate with batch size $1, hidden dimension $2, dropout $3 and L2 $4 on cuda $5, sampling = $6"


if [ $4 -eq 1 ]; then
    if [ $6 -eq 1 ]; then
        CUDA_VISIBLE_DEVICES=$5 python Train.py --train_size 0.98 --valid_size 0.01 \
        --eval_size 0.01 --batch_size $1 --n_epochs 10 --gpu --max_len 500 \
        --embedding_size 136 --hiddens $2 --n_lstms 1 --dropout $3 --L2 --sample \
        --ckpt_file ./models/base_bsz_$1_h_$2_dp_$3_L2_$4_sp_$6_ckp_0.pt 
    else
        CUDA_VISIBLE_DEVICES=$5 python Train.py --train_size 0.98 --valid_size 0.01 \
        --eval_size 0.01 --batch_size $1 --n_epochs 10 --gpu --max_len 500 \
        --embedding_size 136 --hiddens $2 --n_lstms 1 --dropout $3 --L2 \
        --ckpt_file ./models/base_bsz_$1_h_$2_dp_$3_L2_$4_sp_$6_ckp_0.pt 
    fi
else
    if [ $6 -eq 1 ]; then
        CUDA_VISIBLE_DEVICES=$5 python Train.py --train_size 0.98 --valid_size 0.01 \
        --eval_size 0.01 --batch_size $1 --n_epochs 10 --gpu --max_len 500 \
        --embedding_size 136 --hiddens $2 --n_lstms 1 --dropout $3 --sample \
        --ckpt_file ./models/base_bsz_$1_h_$2_dp_$3_L2_$4_sp_$6_ckp_0.pt 
    else
        CUDA_VISIBLE_DEVICES=$5 python Train.py --train_size 0.98 --valid_size 0.01 \
        --eval_size 0.01 --batch_size $1 --n_epochs 10 --gpu --max_len 500 \
        --embedding_size 136 --hiddens $2 --n_lstms 1 --dropout $3 \
        --ckpt_file ./models/base_bsz_$1_h_$2_dp_$3_L2_$4_sp_$6_ckp_0.pt 
    fi
fi
