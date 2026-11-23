# 分阶段开发

## 当前交付：A / 环境与 Random

- [x] 可安装 Python 包、固定依赖、公共环境工厂
- [x] 12 维状态、离散动作、原始奖励、TimeLimit
- [x] 可复现 Random 策略及 4 × 100 局验证实验
- [x] 逐局 CSV、统计汇总、两张图、运行元数据
- [x] 连续三局 Random MP4 及视频对应 CSV
- [x] 环境与评估测试、CI 配置、中文上手文档

## B / DQN（待实现）

- [ ] 在 `src/flappy_bird_rl/agents/dqn/train.py` 中实现训练，读取 `configs/dqn.json` 中的 `model_kwargs`
- [ ] 在 `src/flappy_bird_rl/agents/dqn/agent.py` 中接入公共 `Policy` 协议
- [ ] 使用公共环境与 Monitor；定期保存 checkpoint、验证指标、真实采样步数和训练耗时
- [ ] 比较 exploration_fraction = 0.1 / 0.3 / 0.5
- [ ] 至少三个独立训练种子，统一测试与示范

## C / PPO（待实现）

- [ ] 在 `src/flappy_bird_rl/agents/ppo/train.py` 中实现训练，读取 `configs/ppo.json` 中的 `model_kwargs`
- [ ] 在 `src/flappy_bird_rl/agents/ppo/agent.py` 中接入公共 `Policy` 协议
- [ ] 保存模型、训练诊断和真实采样步数
- [ ] 比较 ent_coef = 0.00 / 0.01 / 0.03
- [ ] 至少三个独立训练种子，统一测试与示范

## D / 奖励改造与综合比较（待实现）

- [ ] 添加奖励包装器，只在训练环境使用
- [ ] 接入已训练模型的统一评估命令
- [ ] 比较 Random / DQN / PPO / Shaped 变体
- [ ] 跨训练种子统计、置信区间、学习曲线和固定环境种子视频

## E / 框架之外的独立高分成果：PQN + 穷举验证（已完成）

- [x] 调研 2023–2026 年的方法与本环境的公开成绩，见 `sota/RESEARCH.md`
- [x] 从零实现 PQN（ICLR 2025），训练只用公共环境工厂，评估走公共 `run_episode`
- [x] 20 个训练种子 + 8 组消融；标准验证/测试协议结果按公共 CSV 合约导出
- [x] 穷举验证器：枚举策略可达的全部游戏状态，给出**精确**死亡概率，而不是抽样估计
- [x] 报告、图表、模型、录像与复现脚本全部在 `sota/` 下，不改动 B/C/D 的占位实现

该成果自成一体，不依赖 DQN/PPO 的后续实现；其逐局 CSV 可以直接与它们的结果拼接比较。

当前 DQN/PPO JSON 明确标注为计划模板，不包含模型。这里的超参数来自项目讨论，仅作实验起点，不宣称最优。
