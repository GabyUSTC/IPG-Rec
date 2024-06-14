import argparse
import torch
from model import SASRec
from utils import *
from tqdm import tqdm
import random
import copy
from simulators import RecSim

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/SASRec_epoch_200.pt', type=str)
parser.add_argument('--n_items', default=50, type=int, help='number of items to be set as target item')
parser.add_argument('--alpha', default=10, type=float, help='alpha in SASRec')
parser.add_argument('--device', default='cuda:4', type=str, help='device to run the model on')
parser.add_argument('--num_users', help='...', default=6034, type=int)
parser.add_argument('--num_items', help='...', default=3533, type=int)
parser.add_argument('--maxlen', default=70, type=int)
parser.add_argument('--hidden_units', default=48, type=int)
parser.add_argument('--num_blocks', default=4, type=int)
parser.add_argument('--num_epochs', default=401, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--env_omega', help='...', default=0.8, type=float)
parser.add_argument('--episode_length', help='...', default=20, type=int)
parser.add_argument('--batch_size', help='...', default=512, type=int)
args = parser.parse_args()

# set random seed
random.seed(2021)
# random sample 50 items as target items
item_list = list(range(args.num_items))
random.shuffle(item_list)
target_items = item_list[:args.n_items]
print(f'target items: {target_items}')

all_hit_ratios = []
all_ratings_avg = []

for target_item in tqdm(target_items):
    dataset = data_partition('omega=0.8_boredom=0.8_k=500_pos')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    model = SASRec(args.num_users, args.num_items, args).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    click_logs_all = torch.zeros(args.num_users, args.episode_length)


    # env = RecSim(device=args.device, episode_length=args.episode_length)
    env = RecSim(device='cpu', episode_length=args.episode_length, env_omega=args.env_omega, boredom_decay=0.8)
    env.load('envs/omega=0.8_boredom=0.8_k=500.pt')

    seq_guidance = []
    for u in range(1, args.num_users + 1):
        seq_u = user_train[u]
        seq_u = [0] * (args.maxlen - len(seq_u)) + seq_u
        seq_guidance.append(seq_u)

    ratings_logs = torch.zeros(args.num_users, args.episode_length)
    interacted = torch.zeros(args.num_users, args.num_items + 1).to(args.device).long()
    batch_num = args.num_users // args.batch_size + 1

    for i in range(args.episode_length):
        recommendations = []
        for batch in range(batch_num):
            if batch == batch_num - 1:
                input_seq = seq_guidance[batch * args.batch_size :]
                interacted_input = interacted[batch * args.batch_size :]
            else:
                input_seq = seq_guidance[batch * args.batch_size : (batch + 1) * args.batch_size]
                interacted_input = interacted[batch * args.batch_size : (batch + 1) * args.batch_size]
            input_seq = copy.deepcopy(input_seq)
            input_seq = np.array(input_seq)
            recommendation = model.next_item_prediction_with_IPG(input_seq, interacted_input, target_item).cpu().numpy().tolist()
            recommendations.extend(recommendation)

        recommendations = torch.LongTensor(recommendations).cpu()
        recommendations[recommendations < 0] = 0

        r_target = env.get_avg_rating(target_item, reduce=False)
        ratings_logs[:, i] = r_target

        obs, total_clicks, done, info = env.step(recommendations.flatten())

        for u in range(args.num_users):
            interacted[u, recommendations[u] + 1] = 1
            if obs['clicks'][u]:
                seq_guidance[u] = [seq_guidance[u][j+1] for j in range(len(seq_guidance[u])-1)] + [recommendations[u].item() + 1]

        click_logs_all[:, i] = obs['clicks']
        hit_ratio = obs['clicks'].sum() / args.num_users
        # print(f"Epoch: {i}, Hit ratio = {hit_ratio :.4f}")
    
    all_hit_ratios.append(click_logs_all.mean().item())
    all_ratings_avg.append(ratings_logs.mean(dim=0))

all_ratings_avg = torch.stack(all_ratings_avg)
print(f'IPG, k=5, hit_ratio={torch.tensor(all_hit_ratios)[:5].mean()}, ratings={(all_ratings_avg[:, 4] - all_ratings_avg[:, 0]).mean()}')
print(f'IPG, k=10, hit_ratio={torch.tensor(all_hit_ratios)[:10].mean()}, ratings={(all_ratings_avg[:, 9] - all_ratings_avg[:, 0]).mean()}')
print(f'IPG, k=15, hit_ratio={torch.tensor(all_hit_ratios)[:15].mean()}, ratings={(all_ratings_avg[:, 14] - all_ratings_avg[:, 0]).mean()}')
print(f'IPG, k=20, hit_ratio={torch.tensor(all_hit_ratios)[:20].mean()}, ratings={(all_ratings_avg[:, 19] - all_ratings_avg[:, 0]).mean()}')