> [cleanrl dqn_atari.py 代码](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L108)
> [cleanrl arati游戏 dqn试验结果](https://docs.cleanrl.dev/rl-algorithms/dqn/)


## 🧱 网络结构详解：从图像输入到 Q 值输出

```python
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),      # Layer 1: Conv
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),     # Layer 2: Conv
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),     # Layer 3: Conv
            nn.ReLU(),
            nn.Flatten(),                       # Layer 4: Flatten
            nn.Linear(3136, 512),               # Layer 5: FC
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),  # Layer 6: Output
        )
```

下面我们一层一层分析：

### ✅ 输入格式：堆叠的灰度帧

通过环境包装器处理后，输入状态是：
- **4 帧灰度图像**
- 每帧大小 `(84, 84)`
- 所以整体输入 shape 为：`(batch_size, 4, 84, 84)`  
  （PyTorch 要求 channel 在前）

> 👉 这里的 `4` 不是 RGB 的三通道，而是**时间上的 4 个连续帧**，用于感知运动方向。

---

### 🔁 第1层：`nn.Conv2d(4, 32, 8, stride=4)`

- 输入通道：4（4 帧）
- 输出通道：32（提取 32 种特征图）
- 卷积核大小：8×8
- 步长：4

计算输出尺寸：
$$
\frac{84 - 8}{4} + 1 = 20
$$
✅ 输出 shape：`(batch_size, 32, 20, 20)`

---

### 🔁 第2层：`nn.Conv2d(32, 64, 4, stride=2)`

- 输入：`(32, 20, 20)`
- 卷积核：4×4，步长 2

输出尺寸：
$$
\frac{20 - 4}{2} + 1 = 9
$$
✅ 输出 shape：`(batch_size, 64, 9, 9)`

---

### 🔁 第3层：`nn.Conv2d(64, 64, 3, stride=1)`

- 输入：`(64, 9, 9)`
- 卷积核：3×3，步长 1

输出尺寸：
$$
\frac{9 - 3}{1} + 1 = 7
$$
✅ 输出 shape：`(batch_size, 64, 7, 7)`

---

### 📦 第4层：`nn.Flatten()`

将所有维度展平成一维向量。

当前体积：`64 × 7 × 7 = 3136`

✅ 展平后 shape：`(batch_size, 3136)`

---

### 💡 第5层：`nn.Linear(3136, 512)`

全连接层，把 3136 维压缩到 512 维，进行高级特征整合。

✅ 输出 shape：`(batch_size, 512)`

---

### 🎯 第6层（输出层）：`nn.Linear(512, env.single_action_space.n)`

这才是最关键的！

- 输入：512 维特征向量
- 输出：等于动作空间的大小

#### 示例：不同游戏的动作数
| 游戏 | 动作数量 |
|------|---------|
| `BreakoutNoFrameskip-v4` | 4（NOOP, FIRE, LEFT, RIGHT） |
| `PongNoFrameskip-v4` | 6（但实际常用 3 或 4） |
| `CartPole-v1` | 2（左推、右推） |

**`QNetwork` 的最终输出是一个向量，表示在当前状态下，每个可能动作的 Q 值估计。**

例如，在 `BreakoutNoFrameskip-v4` 游戏中：
- 动作空间有 4 种：不动、左移、右移、开球（fire）
- 那么网络输出就是一个长度为 4 的向量，形如：

```python
[ 2.1, -0.5, 3.8, 1.0 ]
```

这表示：
- 动作 0（不动）：价值 2.1
- 动作 1（左移）：价值 -0.5
- 动作 2（右移）：价值 3.8 ← 最高 → 智能体会倾向于选择这个动作
- 动作 3（开球）：价值 1.0

🎯 所以，**输出的 shape 是 `[N, num_actions]`**，其中 N 是 batch size。


---

### 🧪 前向传播：`forward` 函数做了什么？

```python
def forward(self, x):
    return self.network(x / 255.0)
```

这里只做了一件事：**将像素值归一化到 [0, 1] 区间**

原始图像像素范围是 0~255，除以 255 后变为 0~1，有利于神经网络训练稳定。

📌 输入 `x` 的 shape：`(batch_size, 4, 84, 84)`  
📌 输出 `q_values` 的 shape：`(batch_size, num_actions)`

---

## ✅ 举个具体例子：Breakout 游戏中的输出

假设你运行的是：

```bash
python dqn_atari.py --env-id BreakoutNoFrameskip-v4
```

那么：
- `env.single_action_space.n == 4`
- 网络输出就是 shape 为 `(1, 4)` 或 `(32, 4)` 的张量（取决于是否批处理）

比如某次前向传播结果：

```python
q_values = tensor([[ 1.2, -0.3,  3.5,  0.8 ]])  # shape: [1, 4]
```

然后代码中这样选动作：

```python
actions = torch.argmax(q_values, dim=1).cpu().numpy()
# → argmax([1.2, -0.3, 3.5, 0.8]) = 2 → 表示“向右移动”
```

---

## 📌 总结：`QNetwork` 输出详解

| 项目 | 内容 |
|------|------|
| **输出类型** | 动作价值函数 $Q(s,a)$ 的估计 |
| **输出形式** | 张量（Tensor），每行对应一个状态的所有动作 Q 值 |
| **输出 shape** | `(batch_size, num_actions)` |
| **数值含义** | 数值越大，表示执行该动作的预期回报越高 |
| **是否带 softmax？** | ❌ 不是概率分布！只是原始得分（logits） |
| **是否需要归一化？** | ❌ 不需要，直接用于 argmax 或 loss 计算 |

---



## 🧭 总览：DQN 核心流程（五步循环）

1. **初始化**：构建 Q 网络、目标网络、经验回放缓冲区
2. **与环境交互**：用 ε-greedy 策略选择动作，收集经验
3. **存储经验**：将 `(s, a, r, s', done)` 存入 Replay Buffer
4. **采样训练**：从 buffer 中随机抽样一批数据进行学习
5. **更新网络**：
   - 用 MSE 损失更新主网络
   - 定期同步或软更新目标网络

下面我们一步步拆解。

---

## 1️⃣ 阶段一：初始化 Setup

```python
# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
)
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

q_network = QNetwork(envs).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
target_network = QNetwork(envs).to(device)
target_network.load_state_dict(q_network.state_dict())

rb = ReplayBuffer(
    args.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    optimize_memory_usage=True,
    handle_timeout_termination=False,
)
```

### ✅ 对应功能说明：

| 组件 | 功能 |
|------|------|
| `SyncVectorEnv` + `make_env` | 创建游戏环境，并应用 Atari 预处理（灰度、缩放、跳帧、堆叠） |
| `QNetwork` | 主网络，用于预测当前状态下的 Q 值 |
| `target_network` | 目标网络，用于稳定 TD 目标计算 |
| `optimizer` | Adam 优化器，负责梯度更新 |
| `ReplayBuffer` | 经验回放缓冲区，保存历史经验供后续复用 |

📌 注意：目标网络初始化时和主网络参数完全相同 → 保证初始目标可信

---

## 2️⃣ 阶段二：与环境交互（Action Selection）

```python
obs, _ = envs.reset(seed=args.seed)
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        q_values = q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
```

### ✅ 对应 DQN 步骤：**ε-greedy 动作选择**

- 使用 `linear_schedule` 实现 ε 的线性衰减：
  - 初始探索率 `start_e=1.0` → 完全随机
  - 最终探索率 `end_e=0.01` → 几乎贪婪
  - 在前 10% 的训练步数内完成衰减

🧠 类比：小孩刚开始乱试，越长大越依赖经验

- 动作选择逻辑：
  - 若随机 < ε → 随机动作（探索）
  - 否则 → 输入当前观测 `obs` 到 Q 网络 → 取最大 Q 值的动作（利用）

> 💡 输入 `obs` 是 shape 为 `(1, 4, 84, 84)` 的 4 帧堆叠图像

---

## 3️⃣ 阶段三：执行动作 & 存储经验

```python
next_obs, rewards, terminations, truncations, infos = envs.step(actions)

# ... 处理 truncation 的 final_observation ...
real_next_obs = next_obs.copy()
for idx, trunc in enumerate(truncations):
    if trunc:
        real_next_obs[idx] = infos["final_observation"][idx]

rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
obs = next_obs  # 更新状态
```

### ✅ 对应 DQN 步骤：**收集经验并存入 Replay Buffer**

- `envs.step(actions)`：执行动作，获得反馈
- `rb.add(...)`：将五元组 `(s, s', a, r, done)` 存入经验池
- 特别处理了 `truncation`（截断）情况，确保 `real_next_obs` 正确设置
- 最后更新 `obs = next_obs`，进入下一时间步

📌 关键点：
> 所有经验都先存起来，不立即训练 → 支持后续**离线批量学习**

---

## 4️⃣ 阶段四：训练阶段（Learning from Experience）

```python
if global_step > args.learning_starts:
    if global_step % args.train_frequency == 0:
        data = rb.sample(args.batch_size)
```

### ✅ 条件判断含义：

- `global_step > args.learning_starts`：预热期过后才开始训练（默认 80,000 步）
  - 目的：让 buffer 先积累足够多的经验
- `global_step % args.train_frequency == 0`：每 4 步训练一次（可调）
  - 节省计算资源，避免频繁更新

- `data = rb.sample(args.batch_size)`：从 replay buffer 中随机抽取一个 batch（默认 32 条经验）

📌 数据结构示例：
```python
data.observations     # shape: [32, 4, 84, 84]
data.actions          # shape: [32, 1]
data.rewards          # shape: [32, 1]
data.next_observations# shape: [32, 4, 84, 84]
data.dones            # shape: [32, 1]
```

---

### ✅ 计算 TD Target（贝尔曼目标）

```python
with torch.no_grad():
    target_max, _ = target_network(data.next_observations).max(dim=1)
    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
```

🎯 这是整个 DQN 的核心公式！

$$
y_t = r_t + \gamma \cdot \max_{a'} Q_{\text{target}}(s_{t+1}, a') \cdot (1 - \text{done}_t)
$$

逐项解释：

| 项 | 含义 |
|-----|--------|
| `target_network(...)` | 用**目标网络**预测下一状态的所有 Q 值 |
| `.max(dim=1)` | 取最大值 → 得到最优动作对应的 Q 值 |
| `args.gamma * target_max` | 加上折扣后的未来价值 |
| `* (1 - data.dones)` | 如果 episode 已结束（done=True），则未来价值为 0 |
| `with torch.no_grad()` | 不记录梯度 → 提高效率且防止反向传播污染目标网络 |

📌 这里使用的是 **Hard Update + 固定目标网络**，不是 Double DQN（但结构已支持扩展）

---

### ✅ 计算当前 Q 值估计

```python
old_val = q_network(data.observations).gather(1, data.actions).squeeze()
```

这一步是在计算：
$$
Q(s_t, a_t; \theta)
$$

具体操作：
- `q_network(...)`：主网络输出每个动作的 Q 值，shape `[32, num_actions]`
- `.gather(1, data.actions)`：选出实际采取的那个动作的 Q 值，shape `[32, 1]`
- `.squeeze()`：压成 `[32]`，便于后续 loss 计算

📌 注意：这里只更新被选中的动作的 Q 值 → 符合 Q-learning 的更新原则

---

### ✅ 计算损失并反向传播

```python
loss = F.mse_loss(td_target, old_val)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

损失函数就是均方误差（MSE）：

$$
\mathcal{L}(\theta) = \mathbb{E}\left[ (y_t - Q(s_t, a_t; \theta))^2 \right]
$$

然后标准的 PyTorch 训练三连：
1. 清除梯度
2. 反向传播
3. 更新参数

📌 日志记录：
```python
writer.add_scalar("losses/td_loss", loss, global_step)
writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
```
- `td_loss`：衡量预测误差大小
- `q_values`：监控是否出现过估计或欠估计

---

## 5️⃣ 阶段五：更新目标网络（Target Network Update）

```python
if global_step % args.target_network_frequency == 0:
    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
        target_network_param.data.copy_(
            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
        )
```

### ✅ 这是 Polyak Soft Update（软更新）

公式为：
$$
\theta_{\text{target}} \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta_{\text{target}}
$$

- 当 `tau=1.0` 时 → 等价于**硬更新**（每隔 1000 步完全复制一次）
- 当 `tau<1.0` 时 → 实现平滑过渡，进一步提升稳定性

📌 默认 `tau=1.0`，所以是每 1000 步做一次硬更新

---

## 📊 日志与评估部分（辅助功能）

```python
if "final_info" in infos:
    for info in infos["final_info"]:
        if info and "episode" in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
```

- 每当一个 episode 结束，记录回合回报（reward sum）
- 用于绘制学习曲线，观察智能体是否在进步

---

## 💾 模型保存与评估

```python
if args.save_model:
    torch.save(q_network.state_dict(), model_path)
    episodic_returns = evaluate(...)  # 使用测试模式运行 10 场游戏
    for idx, ret in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", ret, idx)
```

- 训练结束后保存模型权重
- 调用 `evaluate` 函数，在独立环境中测试性能（ε=0.01，减少随机性）
- 将结果上传至 TensorBoard 和 Hugging Face Hub（可选）

---

## ✅ 总结：DQN 流程与代码映射表

| DQN 步骤 | 对应代码位置 | 关键实现 |
|---------|---------------|-----------|
| 初始化网络 | `q_network`, `target_network` | CNN 架构，Adam 优化器 |
| 初始化经验池 | `ReplayBuffer` | 支持高效采样和 truncation 处理 |
| ε-greedy 动作选择 | `epsilon = linear_schedule(...)` | 线性衰减，鼓励前期探索 |
| 执行动作 | `envs.step(actions)` | 获取新状态和奖励 |
| 存储经验 | `rb.add(...)` | 写入 replay buffer |
| 采样 batch | `rb.sample(batch_size)` | 随机抽样，打破相关性 |
| 计算 TD 目标 | `target_network(...).max()` | 使用目标网络防止自举漂移 |
| 计算当前 Q 值 | `q_network(...).gather(...)` | 提取所选动作的估计值 |
| 计算损失 | `F.mse_loss(td_target, old_val)` | 回归损失驱动学习 |
| 反向传播 | `loss.backward()`, `optimizer.step()` | 更新主网络参数 |
| 更新目标网络 | `copy_ weights every C steps` | 硬更新 / 软更新（Polyak） |
| 日志记录 | `SummaryWriter` | 记录 loss、SPS、return 等指标 |
| 模型评估 | `evaluate(...)` | 独立测试，验证泛化能力 |

---

## 🎯 一句话总结

> 这段 `dqn_atari.py` 代码完整实现了 DQN 的所有核心技术：
>
> **用卷积网络理解像素 → 用经验回放打破序列相关 → 用目标网络稳定学习 → 用 ε-greedy 平衡探索与利用 → 用 MSE 损失逼近最优 Q 函数**
>
> 它不仅是算法的忠实还原，更是工程实践的典范。

---
