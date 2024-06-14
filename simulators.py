import torch
import math
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
    
class RecSim():
    def __init__(self, topic_size=2, num_topics=10, episode_length=300, num_items=3533, num_users=6034, device='cpu', sim_seed=114514, env_offset=0.7, env_slope=10, env_omega=0.8, diversity_penalty=1.0, diversity_threshold=5, recent_items_maxlen=10, short_term_boost=1., boredom_threshold=4, boredom_moving_window=5, bias_penalty=0.1, boredom_decay=0.8):
        self.topic_size = topic_size
        self.num_topics = num_topics
        self.episode_length = episode_length
        self.num_items = num_items
        self.num_users = num_users
        self.device = device
        self.t = 0

        self.sim_seed = sim_seed
        self.rd_gen = torch.Generator(device = device)
        self.rd_gen.manual_seed(sim_seed)
        self._dynamics_random = np.random.RandomState(sim_seed)
        
        # User preference model
        self.offset = env_offset
        self.slope = env_slope
        self.omega = env_omega
        self.diversity_penalty = diversity_penalty
        self.diversity_threshold = diversity_threshold

        # Boredom model
        self.recent_items_maxlen = recent_items_maxlen
        self.short_term_boost = short_term_boost
        self.boredom_thresh = boredom_threshold
        self.boredom_moving_window = boredom_moving_window
        self.bias_penalty = bias_penalty
        self.boredom_decay = boredom_decay
        
    def click_model(self, rels : torch.FloatTensor, comps : torch.LongTensor) -> torch.LongTensor:
        '''
            UBM click model
        '''
        if torch.max(torch.unique(comps, return_counts = True)[1]) >= self.diversity_threshold:
            rels /= self.diversity_penalty ### When too many similar are in the slate, the overall attractiveness of the slate decreases.
        clicks = torch.bernoulli(rels, generator = self.rd_gen)
        return clicks
    
    def reset(self):
        self.t = 0  # Index of the trajectory-wide timestep
        self.clicked_items = [deque([], self.recent_items_maxlen) for _ in range(self.num_users)]
        self.all_clicked_items = [[] for _ in range(self.num_users)]
        
        # bool tensor, if a user is bored with a topic, the corresponding element is True
        self.bored = torch.zeros(self.num_users, self.num_topics, dtype = torch.bool, device = self.device)

        # int tensor, the number of steps that a user is bored with a topic
        self.bored_timeout = 5 * torch.ones(self.num_users, self.num_topics, dtype = torch.long, device = self.device)

        self.user_short_term_comp = torch.randint(self.num_topics, size = (self.num_users, 1), device = self.device, generator = self.rd_gen)

        # init user embeddings
        # self.user_embedd = torch.abs(torch.clamp(0.4 * torch.randn(self.num_users, self.num_topics, self.topic_size,
        #                                                             device = self.device, generator = self.rd_gen), -1, 1))
        self.user_embedd = torch.clamp(0.4 * torch.randn(self.num_users, self.num_topics, self.topic_size,
                                                                    device = self.device, generator = self.rd_gen), -1, 1)
        user_comp_dist = torch.rand(size = (self.num_users, self.num_topics), device = self.device, generator = self.rd_gen).pow(1)
        user_comp_dist /= torch.sum(user_comp_dist, dim=1).unsqueeze(1)
        self.user_embedd *= user_comp_dist.unsqueeze(2)
        self.user_embedd = self.user_embedd.flatten(start_dim = 1)
        user_comp = torch.argmax(torch.stack([torch.linalg.norm(topic, dim = 1)
                                                    for topic in torch.split(self.user_embedd, self.topic_size, dim = 1)]), dim = 0)
        comp_idx = torch.stack([self.topic_size * user_comp , self.topic_size * (user_comp  + 1) - 1])
        self.user_embedd[torch.arange(self.num_users), comp_idx] = torch.abs(self.user_embedd[torch.arange(self.num_users), comp_idx])
        user_norm = torch.linalg.norm(self.user_embedd, dim = 1)
        self.user_embedd /= user_norm.unsqueeze(1)

        self.init_user_embedd = self.user_embedd.clone()

        # init item embeddings
        #### First distribution over topics
        comp_dist = torch.rand(size = (self.num_items, self.num_topics), device = self.device)
        comp_dist /= torch.sum(comp_dist, dim = 1).unsqueeze(1)     ### An item cannot be good in every topic.
        #### Then topic_specific quality and position
        self.item_embedd = torch.clamp(0.4 * torch.randn(self.num_items, self.num_topics, self.topic_size, device = self.device), -1, 1)
        #### Then we can normalize
        self.item_embedd *= comp_dist.unsqueeze(2)
        # For focused embeddings:
        self.item_embedd = self.item_embedd.flatten(start_dim = 1).pow(2)
        self.item_comp = torch.argmax(torch.stack([torch.linalg.norm(topic, dim = 1)
                                                   for topic in torch.split(self.item_embedd, self.topic_size, dim = 1)]), dim = 0)
        # covert the units with the corresponding item_camp of the embeddings to be positive
        comp_idx = torch.stack([self.topic_size * self.item_comp, self.topic_size * (self.item_comp + 1) - 1])
        self.item_embedd[torch.arange(self.num_items), comp_idx] = torch.abs(self.item_embedd[torch.arange(self.num_items), comp_idx])
        embedd_norm = torch.linalg.norm(self.item_embedd, dim = 1)
        self.item_embedd /= embedd_norm.unsqueeze(1)
        
        self.bias = torch.zeros(self.num_users, self.num_items, device=self.device)

    def save_env(self, path):
        # pack all the parameters of the environment
        env_params = {'user_embedd' : self.user_embedd,
                'init_user_embedd' : self.init_user_embedd,
                'item_embedd' : self.item_embedd,
                'item_comp' : self.item_comp,
                'bias' : self.bias,
                'bored' : self.bored,
                'bored_timeout' : self.bored_timeout,
                'user_short_term_comp' : self.user_short_term_comp,
                't' : self.t,
                'clicked_items' : self.clicked_items,
                'all_clicked_items' : self.all_clicked_items
                }
        torch.save(env_params, path)

    def load(self, path):
        # load all the parameters of the environment
        env_params = torch.load(path)
        self.user_embedd = env_params['user_embedd'].to(self.device)
        self.item_embedd = env_params['item_embedd'].to(self.device)
        self.init_user_embedd = env_params['init_user_embedd'].to(self.device)
        self.item_comp = env_params['item_comp'].to(self.device)
        self.bias = env_params['bias'].to(self.device)
        self.bored = env_params['bored'].to(self.device)
        self.bored_timeout = env_params['bored_timeout'].to(self.device)
        self.user_short_term_comp = env_params['user_short_term_comp'].to(self.device)
        self.t = env_params['t']
        self.clicked_items = env_params['clicked_items']
        self.all_clicked_items = env_params['all_clicked_items']

    def step(self, recommendations):
        self.t += 1
        info = {}
        self.bored_timeout -= self.bored.long()
        self.bored = self.bored & (self.bored_timeout != 0)
        self.bored_timeout[self.bored == False] = 5

        info["recommendations"] = recommendations
        info["rec_components"] = self.item_comp[recommendations]

        ## Compute relevances
        item_embedings = self.item_embedd[recommendations]
        bias = self.bias[torch.arange(self.num_users), recommendations]
        score = torch.sum(item_embedings * self.user_embedd, dim=1) - bias
        # norm_score = score / self.max_score
        relevances = 1 / (1 + torch.exp(-(score - self.offset) * self.slope))    ## Rescale relevance

        info["scores"] = score
        info["bored"] = self.bored
        ## Interaction
        clicks = self.click_model(relevances, self.item_comp[recommendations])

        for u in range(self.num_users):
            if clicks[u]:
                self.clicked_items[u].append(recommendations[u])
                self.all_clicked_items[u].append(recommendations[u])
                self.user_embedd[u] = self.omega * self.user_embedd[u] + (1 - self.omega) * self.item_embedd[recommendations[u]]
                if self.bored[u, self.user_short_term_comp[u]] > 0:
                    self.user_short_term_comp[u] = self.item_comp[recommendations[u]]
                self.bias[u, recommendations[u]] += self.bias_penalty
        topic_norm = torch.linalg.norm(self.user_embedd, dim = 1)   
        self.user_embedd /= topic_norm.unsqueeze(1)

        for u in range(self.num_users):
        ## Bored anytime recently items from one topic have been clicked more than boredom_threshold
            if len(self.clicked_items[u]) > 0:
                recent_items = torch.LongTensor(self.clicked_items[u])
                recent_comps = self.item_comp[recent_items]
                recent_comps = torch.histc(recent_comps.float(), bins = self.num_topics, min = 0, max = self.num_topics - 1).long()
                bored_comps = torch.arange(self.num_topics)[recent_comps >= self.boredom_thresh]
                ## Then, these 2 components are put to 0 for boredom_timeout steps
                self.bored[u, bored_comps] = True
                # print("bored", self.bored.nonzero(as_tuple=True)[0])

            bored_comps = torch.nonzero(self.bored[u]).flatten()

            ## Set bored components
            
            for bc in bored_comps:
                if self.item_comp[recommendations[u]] == bc and clicks[u]:
                    ## Hard boredom effect
                    # self.user_embedd[u, self.topic_size * bc : self.topic_size * (bc + 1)] = 0

                    ## Soft boredom effect
                    # if u < 20:
                    #     print(f'user {u} bored comps: {bc}')
                    self.user_embedd[u, self.topic_size * bc : self.topic_size * (bc + 1)] *= self.boredom_decay

            ### Boost short-term component
            self.user_embedd[u, self.topic_size * self.user_short_term_comp[u] : self.topic_size * (self.user_short_term_comp[u] + 1)] *= self.short_term_boost

            ### Normalized user embedding
            self.user_embedd[u] /= self.user_embedd[u].norm()

        info['user_state'] = self.user_embedd
        info["clicks"] = clicks

        ## 6 - Set done and return
        if self.t >= self.episode_length:
            done = True
            info["done"] = True
        else:
            done = False
            info["done"] = False
        
        obs = {'recommendations' : recommendations, 'clicks' : clicks}
        return obs, torch.sum(clicks), done, info
    
    def get_avg_rating(self, target_item, reduce=True):
        bias = self.bias[:, target_item].flatten()
        ratings = torch.matmul(self.user_embedd, self.item_embedd[target_item])
        ratings -= bias
        if reduce:
            return torch.mean(ratings).item()
        else:
            return ratings.cpu()
    
    def get_all_scores(self):
        scores = torch.matmul(self.user_embedd, self.item_embedd.T) - self.bias
        return scores