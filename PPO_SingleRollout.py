import os
import torch
import numpy
import gymnasium


class PolicyNetwork(torch.nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(PolicyNetwork, self).__init__()
		self.network = torch.nn.Sequential(
			torch.nn.Linear(state_dim, hidden_dim),
			torch.nn.ELU(),
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.ELU(),
			torch.nn.Linear(hidden_dim, action_dim),
			torch.nn.Softmax(dim=-1))

	def forward(self, state):
		return self.network(state)

class ValueNetwork(torch.nn.Module):
	def __init__(self, state_dim, hidden_dim):
		super(ValueNetwork, self).__init__()
		self.network = torch.nn.Sequential(
			torch.nn.Linear(state_dim, hidden_dim),
			torch.nn.ELU(),
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.ELU(),
			torch.nn.Linear(hidden_dim, 1))

	def forward(self, state):
		return self.network(state)

class PPO(object):
	def __init__(self, env_name, train):
		self.env_name = env_name
		if train:
			self.env = gymnasium.make(env_name)
		else:
			self.env = gymnasium.make(env_name, render_mode="human")

		self.state_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.n
		self.hidden_dim = 512

		self.policy_learning_rate = 1e-4
		self.value_learning_rate = 1e-4

		self.batch_size = 512
		self.num_epochs = 20

		self.gamma = 0.99
		self.gae_lambda = 0.95
		self.policy_clip_epsilon = 0.2
		self.value_clip_epsilon = 0.2
		self.max_grad_norm = 1.0

		self.c1 = 0.5
		self.c2 = 0.01

		self.kl_threshold = 0.02

		self.policy = PolicyNetwork(self.state_dim, self.hidden_dim, self.action_dim)
		self.value = ValueNetwork(self.state_dim, self.hidden_dim)

		self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_learning_rate)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_learning_rate)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.policy.to(self.device)
		self.value.to(self.device)

	def save_models(self):
		if not os.path.exists(f"models/{self.env_name}"):
			os.makedirs(f"models/{self.env_name}")

		torch.save(self.policy.state_dict(), os.path.join(f"models/{self.env_name}", "policy.pth"))
		torch.save(self.value.state_dict(), os.path.join(f"models/{self.env_name}", "value.pth"))

	def load_models(self):
		policy_path = os.path.join(f"models/{self.env_name}", "policy.pth")
		value_path = os.path.join(f"models/{self.env_name}", "value.pth")
		if os.path.exists(policy_path) and os.path.exists(value_path):
			self.policy.load_state_dict(torch.load(policy_path, weights_only=True))
			self.value.load_state_dict(torch.load(value_path, weights_only=True))
		else:
			raise FileNotFoundError("Model files not found in specified path")

	def choose_action(self, state, train):
		state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		action_probs = self.policy(state_tensor)
		dist = torch.distributions.Categorical(action_probs)
		if train:
			action = dist.sample()
		else:
			action = torch.argmax(action_probs, dim=-1)
		return action.item(), dist.log_prob(action).item()

	def compute_gae(self, rewards, values, next_value, dones):
		advantages = []
		gae = 0
		for t in reversed(range(len(rewards))):
			if t == len(rewards) - 1:
				next_value = next_value
			else:
				next_value = values[t + 1]
			delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
			gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
			advantages.insert(0, gae)
		return advantages

	def train_step(self, states, actions, old_log_probs, advantages, returns, old_values):
		action_probs = self.policy(states)
		dist = torch.distributions.Categorical(action_probs)
		current_log_probs = dist.log_prob(actions)
		entropy = dist.entropy().mean()
		current_values = self.value(states).squeeze()
		old_probs = torch.exp(old_log_probs)

		kl_div = torch.mean(old_probs * (old_log_probs - current_log_probs))

		if kl_div > self.kl_threshold:
			return

		ratios = torch.exp(current_log_probs - old_log_probs)
		surr1 = ratios * advantages
		surr2 = torch.clamp(ratios, 1.0 - self.policy_clip_epsilon, 1.0 + self.policy_clip_epsilon) * advantages
		policy_loss = -torch.min(surr1, surr2).mean()

		value_pred_clipped = old_values + torch.clamp(current_values - old_values, -self.value_clip_epsilon, self.value_clip_epsilon)
		value_losses = (current_values - returns) ** 2
		value_losses_clipped = (value_pred_clipped - returns) ** 2
		value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

		total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

		self.policy_optimizer.zero_grad()
		self.value_optimizer.zero_grad()
		total_loss.backward()

		torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
		torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)

		self.policy_optimizer.step()
		self.value_optimizer.step()

	def collect_trajectory(self):
		states, actions, rewards, log_probs, dones = [], [], [], [], []
		state, _ = self.env.reset()
		episode_rewards = 0
		done = False
		while not done:
			action, log_prob = self.choose_action(state, train=True)
			next_state, reward, terminated, truncated, info = self.env.step(action)
			done = terminated or truncated
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			log_probs.append(log_prob)
			dones.append(done)
			episode_rewards += reward
			state = next_state
		return states, actions, rewards, log_probs, dones, average_rewards

	def train(self, num_episodes):
		best_average_reward = -numpy.inf
		for episode in range(num_episodes):
			states, actions, rewards, log_probs, dones, average_rewards = self.collect_trajectory()
			states_tensor = torch.FloatTensor(states).to(self.device)
			actions_tensor = torch.LongTensor(actions).to(self.device)
			old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)

			with torch.no_grad():
				values = self.value(states_tensor).squeeze()
				old_values = values.clone()
				if dones[-1]:
					next_value = 0
				else:
					next_state = states[-1]
					next_value = self.value(torch.FloatTensor([next_state]).to(self.device)).item()

			values_np = values.cpu().numpy()
			advantages = self.compute_gae(rewards, values_np, next_value, dones)
			returns = [adv + val for adv, val in zip(advantages, values_np)]

			advantages_tensor = torch.FloatTensor(advantages).to(self.device)
			returns_tensor = torch.FloatTensor(returns).to(self.device)

			advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-9)

			indices = numpy.arange(len(states))
			for x in range(self.num_epochs):
				numpy.random.shuffle(indices)
				for start_idx in range(0, len(states), self.batch_size):
					idx = indices[start_idx:start_idx + self.batch_size]

					self.train_step(states_tensor[idx], actions_tensor[idx], old_log_probs_tensor[idx], advantages_tensor[idx], returns_tensor[idx], old_values[idx])

			if numpy.mean(average_rewards) >= best_average_reward:
				best_average_reward = numpy.mean(average_rewards)
				self.save_models()
				print(f"Episode: {episode} Best Average Reward: {best_average_reward}")
				print("Saving Networks!!!")
				with open("output.txt", "a") as file:
					file.write(f"\n Env: {self.env_name} Episode: {episode + 1} Reward: {best_average_reward}")

	def test(self, num_episodes):
		self.load_models()
		for episode in range(num_episodes):
			state, _ = self.env.reset()
			done = False
			while not done:
				action, log_prob = self.choose_action(state, train=False)
				next_state, reward, terminated, truncated, info = self.env.step(action)
				done = terminated or truncated
				if done:
					break
				state = next_state


if __name__ == "__main__":
	# Create and train PPO agent
	# cartpole = PPO(env_name="CartPole-v1", train=True)
	# cartpole.train(num_episodes=100)
	# acrobot = PPO(env_name="Acrobot-v1", train=True)
	# acrobot.train(num_episodes=500)
	# lunarlander = PPO(env_name="LunarLander-v3", train=True)
	# lunarlander.train(num_episodes=500)
	# mountaincar = PPO(env_name="MountainCar-v0", train=True)
	# mountaincar.train(num_episodes=500)

	cartpole = PPO(env_name="CartPole-v1", train=False)
	cartpole.test(num_episodes=5)
	acrobot = PPO(env_name="Acrobot-v1", train=False)
	acrobot.test(num_episodes=5)
	lunarlander = PPO(env_name="LunarLander-v3", train=False)
	lunarlander.test(num_episodes=5)
	lunarlander = PPO(env_name="MountainCar-v0", train=False)
	lunarlander.test(num_episodes=5)
