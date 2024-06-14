import argparse
import torch
from utils import *
from model import SASRec

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='omega=0.8_boredom=0.8_k=500_pos')
parser.add_argument('--num_users', help='...', default=6034, type=int)
parser.add_argument('--num_items', help='...', default=3533, type=int)
parser.add_argument('--device', default='cuda:4', type=str)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--maxlen', default=70, type=int)
parser.add_argument('--hidden_units', default=48, type=int)
parser.add_argument('--num_blocks', default=4, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--lr', default=0.001, type=float)

args = parser.parse_args()
for arg in vars(args):
    print(f'{arg}: {getattr(args, arg)}')

if __name__ == '__main__':
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    epoch_start_idx = 1
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        loss_epoch = 0.
        model.train() # enable model training
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            loss_epoch += loss * len(u)
        print(f"loss in epoch {epoch} : {loss_epoch.item() / len(user_train):.5f}")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'checkpoints/SASRec_epoch_{epoch}.pt')