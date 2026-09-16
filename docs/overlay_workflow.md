# AudioCraft overlay 安全工作流

本文只描述新工作包如何检查补丁。检查器不会改动旧项目，也没有自动应用补丁的代码路径。

## 固定基线与补丁位置

- AudioCraft 必须位于精确提交 `896ec7c47f5e5d1e5aa1e4b260c4405328bf009d`。
- 工作包补丁统一放在 `patches/audiocraft/*.patch`。
- 补丁按文件名的字典序检查。建议使用 `0001-...patch`、`0002-...patch` 这类带序号名称。
- 默认要求 AudioCraft 的 tracked、staged 与 untracked 状态全部为空。

检查命令（可以从任意当前目录执行）：

```bash
python scripts/check_audiocraft_overlay.py \
  --audiocraft-root /absolute/path/to/audiocraft
```

检查器依次执行以下只读判断：确认参数指向 Git 工作树根目录、确认 `HEAD` 精确匹配、确认工作树干净，并按字典序把整组补丁一次性交给 `git apply --check`。这样后一个补丁可以依赖前一个补丁，同时仍不写工作树。任一步失败均返回非零状态；缺少补丁目录或目录中没有 `.patch` 文件也视为失败。

## 查看人工应用命令

需要查看通过检查后的应用命令时，显式增加：

```bash
python scripts/check_audiocraft_overlay.py \
  --audiocraft-root /absolute/path/to/audiocraft \
  --print-apply-command
```

这只会打印每个补丁对应的 `git apply` 命令，不会执行。先审核输出，再由操作者在准备好的 AudioCraft 副本中手工运行。不要在旧项目唯一副本上直接应用；本项目的规范位置是 `${PTC_WORKPACK}/vendor/audiocraft`，从固定基线复制后再应用补丁。这样旧项目保持只读，所有新建/修改内容仍位于一个可复制的 workpack 中。应用后记录 `git status` 与补丁 SHA-256。

## Dirty override 仅用于诊断

`--allow-dirty` 可以绕过默认的干净工作树门禁，但不会改变检查器的只读性质。它只适合定位补丁冲突，不能作为正式训练环境的验收结果。正式应用前应恢复到固定提交的干净工作树，重新运行不带该参数的检查。

## 边界说明

`git apply --check` 证明补丁在当前文件状态下可以被 Git 接受，并不证明训练行为、数值结果或多个相互依赖补丁应用后的组合语义正确。补丁应用后仍需运行工作包单元测试、AudioCraft 两步训练冒烟测试，并保存配置与日志。检查器不会调用 `git reset`、`git checkout`、`git clean` 或任何会修改目标仓库的命令。
