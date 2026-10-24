# 分阶段开发

## 当前交付：A / 环境与 Random

- [x] 可安装 Python 包、固定依赖、公共环境工厂
- [x] 12 维状态、离散动作、原始奖励、TimeLimit
- [x] 可复现 Random 策略及 4 × 100 局验证实验
- [x] 逐局 CSV、统计汇总、两张图、运行元数据
- [x] 连续三局 Random MP4 及视频对应 CSV
- [x] 环境与评估测试、CI 配置、中文上手文档

## B / DQN（待实现）

- [ ] 添加 `src/flappy_bird_rl/train_dqn.py`，读取 `configs/dqn.json` 中的 `model_kwargs`
- [ ] 使用公共环境与 Monitor；定期保存 checkpoint、验证指标、真实采样步数和训练耗时
- [ ] 比较 exploration_fraction = 0.1 / 0.3 / 0.5
- [ ] 至少三个独立训练种子，统一测试与示范

## C / PPO（待实现）

- [ ] 添加 `src/flappy_bird_rl/train_ppo.py`，读取 `configs/ppo.json` 中的 `model_kwargs`
- [ ] 保存模型、训练诊断和真实采样步数
- [ ] 比较 ent_coef = 0.00 / 0.01 / 0.03
- [ ] 至少三个独立训练种子，统一测试与示范

## D / 奖励改造与综合比较（待实现）

- [ ] 添加奖励包装器，只在训练环境使用
- [ ] 接入已训练模型的统一评估命令
- [ ] 比较 Random / DQN / PPO / Shaped 变体
- [ ] 跨训练种子统计、置信区间、学习曲线和固定环境种子视频

当前 DQN/PPO JSON 明确标注为计划模板，不包含模型。这里的超参数来自项目讨论，仅作实验起点，不宣称最优。
