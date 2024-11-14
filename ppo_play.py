from PPO import PPO
import gym
import torch
import numpy as np
import random
import imageio
envs = ["Hopper-v4","InvertedDoublePendulum-v4","Walker2d-v4","Pendulum-v1"]
env_name = envs[0]
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU
random.seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
env = gym.make(env_name,render_mode="rgb_array")
print(env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
K_epochs = 80               # update policy for K epochs in one PPO update
eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
random_seed = 0
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space=True, action_std_init=0.6)
ppo_agent.load(f"PPO_Policy_ver_III/{env_name}/best.pth")
# ppo_agent.load(f"PPO_ver_III_preTrained/{env_name}/PPO_{env_name}_0_0.pth")
state = env.reset()[0]
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
state = torch.FloatTensor(state).to(device)
ep_reward = 0
frames = []
for t in range(1, 100000):
    frame = env.render()
    frames.append(frame)
    print(frame)
    action,action_logprob, state_val = ppo_agent.policy_old.act(state)
    action = action.detach().cpu().numpy().flatten()
    state, reward, done, _,info = env.step(action)
    print(action,reward)
    env.render()
    state = torch.FloatTensor(state).to(device)
    ep_reward += reward
    if done:
        break
print(ep_reward)
# 이미지 리스트를 비디오로 저장
with imageio.get_writer(f'{env_name}.mp4', fps=30) as video:
    for frame in frames:
        video.append_data(frame)

# pip install imageio[ffmpeg]