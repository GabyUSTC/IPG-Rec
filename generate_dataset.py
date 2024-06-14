from simulators import RecSim
import argparse
import torch
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', help='...', default=6034, type=int)
parser.add_argument('--num_items', help='...', default=3533, type=int)
parser.add_argument('--boredom_threshold', help='...', default=4, type=int)
parser.add_argument('--boredom_moving_window', help='...', default=5, type=int)
parser.add_argument('--boredom_decay', help='...', default=0.8, type=int)
parser.add_argument('--env_omega', help='...', default=0.8, type=float)
parser.add_argument('--env_slope', help='...', default=10, type=float)
parser.add_argument('--env_offset', help='...', default=0.8, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--episode_length', help='...', default=100, type=int)
parser.add_argument('--bias_penalty', help='...', default=0.1, type=float)

k = 500 # Top-K for oracle recommendation
args = parser.parse_args()

env = RecSim(num_items=args.num_items, num_users=args.num_users, boredom_moving_window=args.boredom_moving_window,
                boredom_threshold=args.boredom_threshold, env_omega=args.env_omega, env_slope=args.env_slope, env_offset=args.env_offset,
                device=args.device, episode_length=args.episode_length, boredom_decay=args.boredom_decay)

env.reset()
recommended = torch.ones(env.num_users, env.num_items)
all_recommendations, clicks = [], []
for _ in tqdm(range(100)):
    ## Get initial query and/or recommendation
    
    random_recommendations = torch.LongTensor(np.random.choice(list(range(env.num_items)), env.num_users))
    oracle_scores = env.get_all_scores()
    oracle_scores = recommended * oracle_scores
    oracle_recommendation = oracle_scores.max(dim=1)[1]
    _, rec_item = torch.topk(oracle_scores, k=k)
    recommendations = rec_item[torch.arange(args.num_users), torch.randint(k, (args.num_users,))]

    # with the probability of p, the recsys recommend the item with the highest score
    for u in range(args.num_users):
        if np.random.rand() > 0.3:
            recommendations[u] = random_recommendations[u]
        else:
            recommendations[u] = oracle_recommendation[u]
        recommended[u, recommendations[u]] = 0

    obs, _, _, _ = env.step(recommendations)
    ## Store the interaction
    all_recommendations.append(obs["recommendations"])
    clicks.append(obs["clicks"])

dataset = {"recommendations"     : torch.stack(all_recommendations).T,
            "clicks"    : torch.stack(clicks).T}

with open('dataset/omega=0.8_boredom=0.8_k=500_pos.txt', 'w') as f:
    for u in range(args.num_users):
        for j, i in enumerate(dataset['recommendations'][u]):
            if dataset['clicks'][u, j]:
                f.write(f"{u+1} {i+1}\n")
env.save_env(f'envs/omega={args.env_omega}_boredom={args.boredom_decay}_k={k}.pt')
print(f'Average Hit Ratio = {torch.stack(clicks).mean()}')

