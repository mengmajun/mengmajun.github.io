
```python

from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
import torch.nn.functional as F

env_id = "CartPole-v1"
# Create the env
env = gym.make(env_id)

# Create the evaluation env
eval_env = gym.make(env_id, render_mode="human")

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample())  # Get a random observation
print(type(env.observation_space.sample()))

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample())  # Take a random action

device = 'cpu'


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Softmax 的作用是将一组实数（logits）转换成一个概率分布。它对每个元素取指数，然后除以所有元素指数的总和，确保所有输出值都在 [0, 1] 范围内，并且总和为 1。
        # dim=1 表示沿着第 1 个维度（通常是 batch 维度下的特征维度）进行归一化。假设输入是一个形状为 [batch_size, num_actions] 的张量，Softmax 会对每一行（即每一个样本）单独计算其动作概率。
        return F.softmax(x, dim=1)

    def act(self, state):
        # 从numpy数组中创建tensor，pytorch网络的输入必须是tensor， float转为float32，unsqueeze在0维度上增加一个batch维度，比如[4]为[1,4]  to发送到设备比如gpu上
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        # Categorical 是 PyTorch 提供的一个离散概率分布类。
        # 这行代码创建了一个以 probs 为概率质量的分类分布对象 m。例如，如果 probs = [0.7, 0.3]，那么 m 代表一个有 70% 概率选动作 0，30% 概率选动作 1 的分布。
        m = Categorical(probs)
        # 从分布 m 中随机采样一个动作。这是实现随机策略（stochastic policy）的关键，允许智能体进行探索（exploration）。
        action = m.sample()
        # action.item(): 将采样得到的动作张量转换为 Python 的标量（scalar）并返回。.item() 用于从只有一个元素的张量中提取数值。
        # m.log_prob(action): 计算所采取动作的对数概率（log probability）。这在 REINFORCE 等策略梯度算法中至关重要，因为策略梯度定理表明更新方向与 log_prob * reward 成正比。
        return action.item(), m.log_prob(action)


def train_reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # 帮助我们在训练过程中计算得分
    scores_deque = deque(maxlen=100)
    scores = []
    # 伪代码的第3行
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []  # 保存每个步骤t的对数概率
        rewards = []  # 保存每个步骤的奖励
        state, info = env.reset()
        # 伪代码的第4行
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        # 记录每个回合的总得分
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # 伪代码的第6行：计算回报（return）
        # 存储记录每个时间步的预期回报
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # 在每个时间步计算折扣回报，
        # 其定义为：
        #      时间t的gamma折扣回报 (G_t) 加上 时间t的奖励
        #
        # 时间复杂度为 O(N)，其中 N 是时间步的数量
        # （这个关于折扣回报 G_t 的定义遵循了 Sutton & Barto 2017 第二版草稿第44页上的定义）
        # G_t = r_(t+1) + r_(t+2) + ...

        # 根据这个公式，每个时间步t的回报可以通过重用已计算出的未来回报 G_(t+1) 来计算当前的 G_t
        # G_t = r_(t+1) + gamma * G_(t+1)
        # G_(t-1) = r_t + gamma * G_t
        # （这遵循了动态规划的方法，通过记忆化解决方案来避免重复计算）

        # 这是正确的，因为上述公式等价于（另见 Sutton & Barto 2017 第二版草稿第46页）
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...

        ## 基于以上，我们按如下方式计算时间步t的回报：
        #               gamma * G_(t+1) + reward[t]
        #
        ## 我们从最后一个时间步开始向前计算，以便利用上述公式
        ## 并避免从前往后计算时可能需要的冗余计算。

        ## 因此，队列 "returns" 将以时间顺序存放值，从 t=0 到 t=n_steps, 也就是最终结果，索引0位置就是时间t=0的预期回报
        ## 这要归功于 appendleft() 函数，它可以在常数时间 O(1) 内将元素添加到位置0
        ## 而普通的 Python 列表则需要 O(N) 的时间。

        for t in range(n_steps)[::-1]:  # 从最后一个时间步往前算，这样可以避免重复计算
            # 对于时间步t的预期回报是 G_t = r_(t+1) + gamma*G_(t+1)
            # returns[0] 表示的是 t+1 时间步开始的预期回报G_(t+1), 因为我们是从后往前算，第一个位置是最新的t+1时间步的预期回报
            disc_return_t_1 = returns[0] if len(returns) > 0 else 0  # else 0 是因为最后一个时间步没有未来奖励
            # rewards[t] 表示时间步t执行动作后转移到下一个状态state_(t+1)获得的奖励，记作r_(t+1)
            returns.appendleft(gamma * disc_return_t_1 + rewards[t])

        ## 对回报进行标准化，以使训练更加稳定
        eps = np.finfo(np.float32).eps.item()
        ## eps 是能表示的最小浮点数，添加到回报的标准差中是为了避免数值不稳定
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # 伪代码的第7行：
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            # 负号的存在，是为了把“最大化奖励”的目标，转换成一个可以被Adam优化器（默认执行梯度下降）处理的“最小化损失”问题
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # 伪代码的第8行：PyTorch 偏好梯度下降
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))

    return scores


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, _, _ = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 500,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# Create policy and place it to the device
cartpole_policy = Policy(
    cartpole_hyperparameters["state_space"],
    cartpole_hyperparameters["action_space"],
    cartpole_hyperparameters["h_size"],
).to(device)

cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

scores = train_reinforce(
    cartpole_policy,
    cartpole_optimizer,
    cartpole_hyperparameters["n_training_episodes"],
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["gamma"],
    100,
)

mean_reward, std_reward = evaluate_agent(
    eval_env, cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"], cartpole_policy
)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
env.close()

```
