# CLAUDE.md

本文件为Claude Code (claude.ai/code) 在处理此代码库时提供指导。

## 项目概述

这是一个基于AlphaZero风格强化学习构建的中国象棋AI系统。该系统包括神经网络训练、基于MCTS的游戏策略以及与象棋引擎的集成。

## 核心架构

### 主要组件
- **AlphaZero/**: 核心AI实现，包含神经网络、MCTS和模型管理
  - `Enhanced_AI_Player.py`: 主要AI玩家，带有优化的MCTS算法
  - `ModelManager.py`: 神经网络模型加载和管理  
  - `ZobristHash.py`: 位置哈希用于置换表
  - `MemoryManager.py`: 内存优化和搜索树清理
- **chess/**: 游戏逻辑和棋盘表示
  - `static_env.py`: 优化的棋盘状态表示
  - `chessboard.py`: 棋盘可视化和游戏规则
- **game/**: GUI和游戏界面
  - `play.py`: 人机对战的主要游戏界面
  - `PlayToSelf.py`: AI自对弈用于训练数据生成
- **worker/**: 训练和数据生成工作流
  - `Train.py`: 神经网络训练管道
  - `TrainDataGenerater.py`: 自对弈数据生成
- **utils/**: 数据输入输出、日志记录和引擎集成工具

### 配置系统
- `config.py`: 基础配置类
- `config_enhanced.py`: 增强配置，包含MCTS优化，包括UCB1-TUNED、RAVE、渐进解锁和置换表

## 常用命令

在项目根目录下运行命令：

```bash
# 通过自对弈生成训练数据
python sources/main.py generate_data

# 训练神经网络模型
python sources/main.py train

# 与AI对战（人机对战）
python sources/main.py play

# AI自对弈用于评估
python sources/main.py play_to_self

# 强制AI先手
python sources/main.py play --ai-move-first

# 使用UCCI协议进行引擎通信
python sources/main.py generate_data --ucci
```

## 关键文件位置

- 模型: `data/model/` (最佳模型权重和配置)
- 训练数据: `data/train_data/` (自对弈游戏记录)
- 游戏记录: `data/play_record/` (PGN格式)
- 象棋引擎: `data/Engine/` (Pikafish引擎二进制文件)
- 开局库: `data/Books/` (开局库数据库)
- 日志: `logs/` (主要、训练、对战活动的独立日志)

## 引擎集成

系统根据CPU能力自动选择合适的Pikafish引擎二进制文件：
- Windows: `pikafish-avx512.exe` > `pikafish-vnni512.exe` > 后备版本
- macOS/Linux: `pikafish` (通用二进制文件)

## 开发说明

- 系统使用增强的MCTS，包含多种优化（UCB1-TUNED、RAVE、置换表）
- 内存管理至关重要 - MemoryManager处理搜索树清理
- 训练需要大量计算资源并生成大型数据集
- 游戏记录以内部JSON格式和标准PGN格式保存
- GUI使用tkinter，带有自定义棋子图像和主题