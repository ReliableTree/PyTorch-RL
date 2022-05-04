import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

from metaworld.policies.sawyer_bin_picking_v2_policy import SawyerBinPickingV2Policy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

#print(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)

parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
#env = gym.make(args.env_name)
env_creator = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['bin-picking-v2-goal-observable']
env = env_creator()
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)
policy_net, _, _ = pickle.load(open(args.model_path, "rb"))
running_state.fix = True



def main_loop():
    expert_traj = []

    gt_policy = SawyerBinPickingV2Policy()
    num_steps = 0

    for i_episode in count():
        env = env_creator()
        state = env.reset()
        state_save = running_state(state)
        reward_episode = 0
        try_traj = []
        for t in range(501):
            #state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            #action = policy_net(state_var)[0][0].detach().numpy()
            action = gt_policy.get_action(state)
            # choose stochastic action
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action_save = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            done = (reward >= 10)
            next_state_save = running_state(next_state)
            reward_episode += reward
            num_steps += 1

            try_traj.append(np.hstack([state_save, action_save]))

            if args.render:
                env.render()
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state
            state_save = next_state_save
        if done:
            expert_traj += try_traj
        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))
        print(num_steps)
        if num_steps >= args.max_expert_state_num:
            break
    return expert_traj


expert_traj = main_loop()
expert_traj = np.stack(expert_traj)
pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(), 'expert_traj/{}_expert_traj.p'.format(args.env_name)), 'wb'))
