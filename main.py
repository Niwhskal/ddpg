# DDPG implementation from the paper Continuous Control with DRL learning (https://arxiv.org/pdf/1509.02971.pdf)
# Author: Lakshwin Shreesha M K 
# Date: 31st March 2022
# github: niwhskal


import os
import jax
import gym
import rlax
import optax
import argparse
import functools
import haiku as hk
import numpy as np
import jax.numpy as jnp
from optax import OptState
from typing import Any, Tuple
import matplotlib.pyplot as plt

OptState = Any

#--------------------------------------------------
#--------------------------------------------------

# Organization of this implementation is as follows:

# 1. Creating the Actor and Critic network's
# 2. Implementing vanilla experience replay
# 3. Creating the DDPG agent.
# 4. Training loop
# 5. Evaluation loop

#--------------------------------------------------
#--------------------------------------------------



#--------------------------------------------------
# Section 1. Creating Actor and Critic neural networks as modular classes.
#--------------------------------------------------
class Actor(hk.Module):
    # This is the policy network. It's a 3 layered neural network, which takes in a low-dimensional state vector as input, passes it thought two hidden layers and outputs an action.

    def __init__(self, hidden_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim 
        self.action_dim = action_dim
        self.max_action = max_action

    def __call__(self, state: np.ndarray) -> jnp.DeviceArray:
        actor_net = hk.Sequential([
        hk.Flatten(),
        hk.Linear(self.hidden_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
        jax.nn.relu,
        hk.Linear(self.hidden_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
        jax.nn.relu,
        hk.Linear(self.action_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform'))
    ])
        return jnp.tanh(actor_net(state)) * self.max_action  #tanh has a range of (-1, 1) which makes it ideal to output a continuous action. It is muliplied by max_action to scale it to the range of actions accepted by the environment.


class Critic(hk.Module):

    #This is the Q-network. It is once again a 3 layered NN, which takes in a state-action pair, and outputs an updated Q value of the state-action pair. 


    def __init__(self, hidden_dim: int, action_dim:int):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def __call__(self, state_action: np.ndarray) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        critic_net = hk.Sequential([
        hk.Flatten(),
        hk.Linear(self.hidden_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
        jax.nn.relu,
        hk.Linear(self.hidden_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')),
        jax.nn.relu,
        hk.Linear(self.action_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform'))
    ])

        return critic_net(state_action)

#--------------------------------------------------
# Section 2. Implementing vanilla Experience replay
#--------------------------------------------------

class ReplayBuffer(object):
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size
        self.ptr = 0 #pointer to add experiences
        self.size = 0

        #initialization
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float
    ) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.not_done[self.ptr] = 1. - done #not done flag is useful while calculating critic loss.

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, rng: jax.numpy.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        ind = jax.random.randint(rng, (batch_size, ), 0, self.size) #sample a random batch of samples and return them

        return (self.state[ind], 
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.not_done[ind])



#--------------------------------------------------
# Section 3. Creating the DDPG agent
#--------------------------------------------------


#This is a outside-class function to update target network parameters.
@jax.jit
def soft_update(target_params: hk.Params, online_params: hk.Params, tau: float = 0.005) -> hk.Params:
    return jax.tree_multimap(lambda x, y: (1 - tau) * x + tau * y, target_params, online_params)


# Class for the DDPG agent
class Agent(object):
    def __init__(
            self,
            hidden_dim: int,
            action_dim: int,
            max_action: float,
            lr: float,              # A lr=1e-03 is used for both actor and critic networks (also for their target nets as well)
            discount: float,        # The discount factor to calculate q value using the critic(policy) network
            noise_clip: float,      # Ensuring that a lot of noise is not added
            policy_noise: float,    # How much noise to add to the actor networks outputs
            policy_freq: int,       # How often the actor and target networks need to be updated, set to 1 for ddpg, but training is much more stable when 2 or 3 is used.
            actor_rng: jnp.ndarray, # Random seed for actor
            critic_rng: jnp.ndarray, #Random seed for critic
            sample_state: np.ndarray # initial state to initialize actor and critic networks
    ):
        self.discount = discount
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.max_action = max_action

        # Converting the Actor() class to a pure function
        self.actor = hk.without_apply_rng(hk.transform(lambda x: Actor(hidden_dim, action_dim, self.max_action)(x)))

        # Initializing optimizer for the actor
        self.actor_opt = optax.adam(lr)

        # Converting the Critic() class to a pure function
        self.critic = hk.without_apply_rng(hk.transform(lambda x: Critic(hidden_dim, action_dim)(x)))

        #Initializing the critic optimizer
        self.critic_opt = optax.adam(lr)

        #Initializing the actor network (and it's target) using a seed value and sample_inputs
        self.actor_params = self.target_actor_params = self.actor.init(actor_rng, sample_state)
        #Linking actor_params to actor optimizer
        self.actor_opt_state = self.actor_opt.init(self.actor_params)

        # Simple test to check if all is good. We get a sample action as a result
        action = self.actor.apply(self.actor_params, sample_state)

        #Initializing the critic network (and it's target) using a seed value and sample_inputs
        self.critic_params = self.target_critic_params = self.critic.init(critic_rng, jnp.concatenate((sample_state, action), 0))
        self.critic_opt_state = self.critic_opt.init(self.critic_params)

        #used to monitor update frequency of the critic
        self.updates = 0

    def update(self, replay_buffer: ReplayBuffer, batch_size: int, rng: jnp.ndarray) -> None:

        self.updates += 1       #Critic is updated in every iteration, so this counter is incremented
        replay_rand, critic_rand = jax.random.split(rng) # Get a new seed

        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size, replay_rand)             #Sample an experience

        self.critic_params, self.critic_opt_state = self.update_critic(self.critic_params, self.target_critic_params,
        self.target_actor_params, self.critic_opt_state,
        state, action, next_state, reward, not_done,
        critic_rand)            #Update the critic

        if self.updates % self.policy_freq == 0:  #Update actor and target networks with a delay. Setting it to 2 or 3 greatly increases stability during training. 

            self.actor_params, self.actor_opt_state = self.update_actor(self.actor_params, self.critic_params,
            self.actor_opt_state, state)    # Update the actor
            self.target_actor_params = soft_update(self.target_actor_params, self.actor_params)                    #Update the actor_target to those of the actors parameters
            self.target_critic_params = soft_update(self.target_critic_params, self.critic_params)                  #Update the critic_target to those of the critic's parameters

    @functools.partial(jax.jit, static_argnums=0)
    def critic_1(
            self,
            critic_params: hk.Params,
            state_action: np.ndarray
    ) -> jnp.DeviceArray:
        return self.critic.apply(critic_params, state_action)   # THis is a helper function to caluclate outputs from the critic

    @functools.partial(jax.jit, static_argnums=0)
    def actor_loss(
            self,
            actor_params: hk.Params,
            critic_params: hk.Params,
            state: np.ndarray
    ) -> jnp.DeviceArray:
        action = self.actor.apply(actor_params, state) #Get action from policy network
        return - jnp.mean(self.critic_1(critic_params, jnp.concatenate((state, action), 1)))   #concatenate this action with the state and get it's value from the critic. This gives us the expected return, since we need to maximize it, we execute it's equivalent by minimizing the -ve of it.

    @functools.partial(jax.jit, static_argnums=0)
    def update_actor(
            self,
            actor_params: hk.Params,
            critic_params: hk.Params,
            actor_opt_state: OptState,
            state: np.ndarray
    ) -> Tuple[hk.Params, OptState]:
        _, gradient = jax.value_and_grad(self.actor_loss)(actor_params, critic_params, state)
        updates, opt_state = self.actor_opt.update(gradient, actor_opt_state)
        new_params = optax.apply_updates(actor_params, updates)
        return new_params, opt_state    #updating actor parameters and it's optimizer state

    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss(
            self,
            critic_params: hk.Params,
            target_critic_params: hk.Params,
            target_actor_params: hk.Params,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            reward: np.ndarray,
            not_done: np.ndarray,
            rng: jnp.ndarray
    ) -> jnp.DeviceArray:


        #Note: The paper mentions usage of the Ornstein-Uhlenbeck process to add temporally correlated noise. I was not able to get good results using the OU process. I therefore manually inject noise using a guassian 

        noise = (
                jax.random.normal(rng, shape=action.shape) * self.policy_noise
        ).clip(-self.noise_clip, self.noise_clip) # Ensure that the noise is not too much by clipping

        next_action = (
                self.actor.apply(target_actor_params, next_state) + noise
        ).clip(-self.max_action, self.max_action)   # Ensure sure the noisy action is within the valid bounds.

        next_q = self.critic.apply(target_critic_params, jnp.concatenate((next_state, next_action), 1))  #Calculate q-value using the target network for future state and action
        target_q = jax.lax.stop_gradient(reward + self.discount * next_q * not_done) # calculate the advantage function, which in case of DDPG is nothing but the bellman equation.
        #jax.lax.stop_gradient is used to not let the gradient flow through this operation. it's not required
        q_1= self.critic.apply(critic_params, jnp.concatenate((state, action), 1))

        return jnp.mean(rlax.l2_loss(q_1, target_q))  #Caluclate the l2 loss and return it's mean value as the value of a state-action pair

    @functools.partial(jax.jit, static_argnums=0)
    def update_critic(
            self,
            critic_params: hk.Params,
            target_critic_params: hk.Params,
            target_actor_params: hk.Params,
            critic_opt_state: OptState,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            reward: np.ndarray,
            not_done: np.ndarray,
            rng: jnp.ndarray
    ) -> Tuple[hk.Params, OptState]:
        _, gradient = jax.value_and_grad(self.critic_loss)(critic_params, target_critic_params, target_actor_params, state, action, next_state, reward, not_done, rng)
        updates, opt_state = self.critic_opt.update(gradient, critic_opt_state)
        new_params = optax.apply_updates(critic_params, updates)
        return new_params, opt_state  #Update critic network and it's optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def policy(self, actor_params: hk.Params, state: np.ndarray) -> jnp.DeviceArray:
        #This is just a function to get the action, given a state. Helps us get actions.
        return self.actor.apply(actor_params, state)

#--------------------------------------------------
#Parsing all arguments
#--------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="InvertedPendulum-v2") 
    parser.add_argument("--seed", type=int, default =123)
    parser.add_argument("--eval_seed", default = 384234, type=int)
    parser.add_argument("--start_timesteps", default=25000, type=int)
    parser.add_argument("--run_mode", default='train', type=str)  
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--replay_size", default=200000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--policy_freq", default=2, type=int) 
    parser.add_argument("--save_dir", type =str, default='./results/')
    parser.add_argument("--plot_rewards", default= True, type=bool)
    args = parser.parse_args()

    return args

#--------------------------------------------------
#--------------------------------------------------

#--------------------------------------------------
# Section 4. Training loop
#--------------------------------------------------

def train(args):

    best_reward = -np.inf   # Helps keep track of the best_reward so that we can save model_parameters

    env = gym.make(args.env)   # Make gym env
    env.seed(args.seed)

    # Get environment dimensions and other info 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256
    max_action = float(env.action_space.high[0])

    # Initialize environemnt
    state, done = env.reset(), False
    episode_reward = 0
    episode_timestep = 0
    episode_num = 0

    # Used to keep track of episodic rewards so that we can plot them later
    episode_reward_tracker = []

    # Initialize seed
    rng = jax.random.PRNGKey(args.seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    # Initialize ddpg agent
    ddpg_agent = Agent(hidden_dim, action_dim, max_action, args.lr, args.discount, args.noise_clip, args.policy_noise, args.policy_freq, actor_key, critic_key, state)

    # Initialize replay buffer
    rb = ReplayBuffer(state_dim, action_dim, max_size = args.replay_size)


    # Main iteration loop
    for t in range(int(args.max_timesteps)):
        episode_timestep += 1       

        if t < args.start_timesteps:                # this condition is set so that initially, we populate the replay buffer with enough examples to start training.
            action = env.action_space.sample()

        else:
            rng, noise_rng = jax.random.split(rng)
            action = (
                    ddpg_agent.policy(ddpg_agent.actor_params, state)
                    + jax.random.normal(noise_rng, (action_dim, )) * max_action * args.expl_noise
            ).clip(-max_action, max_action)         # I opted not to use Ornstein-Uhlenbeck (OU) process to generate noise, instead I add random noise scaled to the action space of the env. I found this to be much better than the OU process.

        next_state, reward, done, _ = env.step(action)  #take a step forward

        done_bool = float(done) if episode_timestep < env._max_episode_steps else 0   # quit when an episode exceeds the maximum number of timesteps allowed to run in the environment

        rb.add(state, action, reward, next_state, done_bool)  # Populate the replay buffer

        state = next_state
        episode_reward+= reward

        if t >= args.start_timesteps:               # Once we've populated the replay buffer, we can begin updating our agent
            rng, update_rng = jax.random.split(rng)
            ddpg_agent.update(rb, args.batch_size, update_rng)

        if episode_reward >= best_reward:           # Model saving: We save when an episode gets a good reward, >= is set because, when rewards are equal, behaviour of the last episode looked aesthetically pleasing :) 

            best_reward = episode_reward
            if not os.path.exists(os.path.join(args.save_dir, args.env)):  # Check if save_dir exits, if not create it.
                os.makedirs(args.save_dir + args.env)
        
            print('Saving {} parameters at {}\n'.format(args.env, args.save_dir)) 

            # Save model parameters as .npy files
            
            jnp.save(os.path.join(args.save_dir, args.env) + "/{}_actor.npy".format(args.env), ddpg_agent.actor_params)
            jnp.save(os.path.join(args.save_dir, args.env) + "/{}_critic.npy".format(args.env), ddpg_agent.critic_params)
            jnp.save(os.path.join(args.save_dir, args.env) + "/{}_actor_target.npy".format(args.env), ddpg_agent.target_actor_params)
            jnp.save(os.path.join(args.save_dir, args.env) + "/{}_crit_target.npy".format(args.env), ddpg_agent.target_critic_params)


        if done:     # Print the current episode's data and reset the environment
            print('Timesteps: {}, Episode_num {}, EpisodeTS: {}, EpisodeReward: {}'.format(t+1, episode_num +1, episode_timestep, episode_reward))

            if t >=args.start_timesteps:                    # We plot rewards of only those episodes for which an action was taken by our agent (and not sampled randomly)
                episode_reward_tracker.append(episode_reward)

            state, done = env.reset(), False
            episode_reward = 0
            episode_timestep = 0
            episode_num += 1

    
    # plotting function
    if args.plot_rewards:
        plt.plot(episode_reward_tracker, label = 'Reward')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(args.save_dir, args.env) + str(args.env) + '_plot')
        


#--------------------------------------------------
# Section 5. Evaluation Loop
#--------------------------------------------------
def eval(args):

    env = gym.make(args.env)
    env.seed(args.seed)

    # Get environment characteristics
    state_dim = env.observation_space.shape[0]
    hidden_dim = 256
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initalize environment
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0


    # set seed value
    rng = jax.random.PRNGKey(args.eval_seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    # Create the agent
    ddpg_agent = Agent(hidden_dim, action_dim, max_action, args.lr, args.discount, args.noise_clip, args.policy_noise, args.policy_freq, actor_key, critic_key, state)

    # Load pre-trained weights for actor and critic networks from save_dir
    act_params = jnp.load(os.path.join(args.save_dir, args.env) + "/{}_actor.npy".format(args.env), allow_pickle=True)
    crit_params = jnp.load(os.path.join(args.save_dir, args.env) + "/{}_critic.npy".format(args.env), allow_pickle = True)
    target_actor = jnp.load(os.path.join(args.save_dir, args.env) + "/{}_actor_target.npy".format(args.env), allow_pickle=True)
    target_crit = jnp.load(os.path.join(args.save_dir, args.env) + "/{}_crit_target.npy".format(args.env) ,allow_pickle=True)

    ddpg_agent.actor_params = act_params.item()
    ddpg_agent.critic_params = crit_params.item()
    ddpg_agent.target_actor_params = target_actor.item()
    ddpg_agent.target_critic_params = target_crit.item()

    # render evaluations for a number of timesteps

    for i in range(50000):  # this number is set arbitrarily. The environemnt terminates automatically as soon as a max_environemnt_steps counter is reached
        env.render()    # show a frame
        action = ddpg_agent.policy(ddpg_agent.actor_params, state) # take an action according to pre-trained agent
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_timesteps += 1

        if done: # print episode data and reset the environment
            state, done = env.reset(), False
            print('Terminated after {} timesteps with avg reward = {}'.format(episode_timesteps+1, episode_reward))
            episode_timesteps = 0
            episode_reward = 0

    env.close()


#--------------------------------------------------
# Section 6. Script run
#--------------------------------------------------

if __name__ == '__main__':

    args = parse_arguments()

    # Note: in order to evaluate, pre-trained weights must be saved before in save_dir/"env_name"

    if args.run_mode == 'eval':
        print('Evaluting...')
        eval(args)

    else:                 # We train and evaluate by defualt
        print('Training...\n')
        train(args)
        print('Done Training; Evalutaing... \n')
        eval(args)