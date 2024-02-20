import gym
from elegantrl.agents import AgentPPO
from elegantrl.train.config import get_gym_env_args
import torch
import time

# Specify the environment and agent information
env_name = "BipedalWalkerHardcore"
env = gym.make(env_name)
env_args = get_gym_env_args(env, if_print=False)
net_dims = [128, 64]
agent_class = AgentPPO
actor_path = "BipedalWalkerHardcore_PPO_0/act.pth"  # Replace with the actual path to your trained agent

# Create the agent
state_dim = env_args['state_dim']
action_dim = env_args['action_dim']
agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
actor = agent.act
del agent  # Remove the agent after obtaining the actor

# Load the trained actor
print(f"| render and load actor from: {actor_path}")
checkpoint = torch.load(actor_path, map_location=lambda storage, loc: storage)
actor.load_state_dict(checkpoint.state_dict())

# Reset the environment
observation = env.reset()

# Render the agent
for _ in range(1000):
    env.render()
    print("Observation:", observation)
    action = actor(torch.Tensor(observation).to("cpu")).detach().cpu().numpy()
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()

# Pause for a few seconds before closing the environment
input("Press enter to close")
env.close()
