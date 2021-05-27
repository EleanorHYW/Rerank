import os
import json
import torch
import argparse
from RerankingDataset import RerankingDataset
from torch.utils.data import DataLoader
import numpy as np
from PointerNet import PointerNet

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
parser.add_argument('--n_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=False, action='store_true', help='Bidirectional')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--history_len', default=5, type=int, help='history sequence length')

parser.add_argument('--n_test', type=int, default=200, help='Number of  dataset')
# parser.add_argument('--output_dir', type=str, default="./output", help='')
parser.add_argument('--ckpt_file', type=str, default='./models/checkpoint_0.pkl', help='model file path')
parser.add_argument('--max_len', default=50, type=int, help='max item sequence length')
parser.add_argument('--search', type=str, default='greedy', help='greedy/beam')
parser.add_argument('--beam_width', type=int, default=12, help='beam search width')

args = parser.parse_args()
print(args)

# def beam_search(model, batch_x, max_len=10, k=3):
#     return

def greedy_infer(model, test_input, search='greedy', beam_width=5):
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(len(test_input)):
            if search == "greedy":
                input = test_input[i]['Feeds']
                history = test_input[i]['History']
                input_ = input.view(-1, input.size(0), input.size(1))
                history_ = history.view(-1, history.size(0), history.size(1))
                logits, ids = model(input_, history_)
            # elif search == "beam":
                # result = beam_search(model, n=batch_x, max_len, k=beam_width)
            else:
                raise NameError("Unknown search method")
            results.extend(ids)
    print(results)
    print("Done!")

def greedy_infer_batch(model, test_dataloader, search='greedy', beam_width=5):
    results = []
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(test_dataloader):
            if search == "greedy":
                logits, ids = model(sample['Feeds'], sample['History'])
            # elif search == "beam":
                # result = beam_search(model, x_stncs, x_ids, tgt_vocab, k=beam_width)
            else:
                raise NameError("Unknown search method")
            results.append(ids)
    print(results)
    print("Done!")

def main():
    if not os.path.exists(args.ckpt_file):
        raise FileNotFoundError("model file not found")

    dataset = RerankingDataset(args.n_test, args.max_len, args.history_len)

    model = PointerNet(args.embedding_size,
                       args.hiddens,
                       args.n_lstms,
                       args.dropout,
                       args.bidir)

    saved_state = torch.load(args.ckpt_file)
    model.load_state_dict(saved_state['state_dict'])
    print('Load model parameters from %s' % args.ckpt_file)

    # infer one by one
    greedy_infer(model, dataset)

    # infer by batch
    # test_dataloader = DataLoader(dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=True,
    #                               num_workers=4,
    #                               drop_last=True)
    #
    # greedy_infer_batch(model, test_dataloader, search='greedy')


if __name__ == '__main__':
    main()