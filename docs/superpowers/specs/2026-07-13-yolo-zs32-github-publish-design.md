# YOLO ZS32 GitHub 发布设计

## 目标

将当前基于 Ultralytics 的 ZS32 训练代码发布到
`git@github.com:wjstx0425/yolo_zs32.git`，供训练服务器拉取，同时避免把本地数据集、训练输出和模型权重提交到 GitHub。

## 发布边界

- 提交 C789/ZS32 示例代码、离线测试、训练设计文档和目录记忆。
- 忽略仓库根目录的 `dataset/`。
- 继续使用现有规则忽略 `runs/`、`weights/` 和常见模型权重格式。
- 不修改或删除任何本地数据、训练结果及权重。

## Git 远端与分支

- 保留 `origin` 指向 `https://github.com/ultralytics/ultralytics.git`，用于获取上游更新。
- 新增 `yolo-zs32` 远端，指向 `git@github.com:wjstx0425/yolo_zs32.git`。
- 当前本地开发分支保持为 `c789-defect-yolo`。
- 将当前提交发布为目标仓库的 `main`，并让本地分支跟踪 `yolo-zs32/main`。

## 验证

发布前执行以下检查：

1. Git 暂存区中不含 `dataset/`、`runs/` 或模型权重。
2. 暂存文件中不存在超过 GitHub 单文件限制的大文件。
3. 执行 C789 离线测试和 Python 语法检查。
4. 推送后通过 `git ls-remote` 验证目标仓库 `main` 指向本地提交。

## 服务器训练数据流

服务器通过 GitHub 克隆代码；数据集不经过 GitHub，而使用 `rsync`、共享存储或对象存储单独传输。服务器上的数据路径通过训练命令或数据 YAML 显式指定，不在代码仓库中固化大文件。
