
# 8.5 沙箱生命周期

┌─────────────────────────────────────────────────────────────────┐
│                     Sandbox 生命周期流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  lazy_init=True (默认)                                           │
│  ───────────────────                                             │
│                                                                  │
│   Agent 调用 ──→ before_agent() ──→ 跳过获取                      │
│       │                                                          │
│       ▼                                                          │
│   工具调用 ──→ 检查沙箱状态                                        │
│       │                                                          │
│       ├── 无沙箱 ──→ acquire() ──→ 创建/获取沙箱                   │
│       │                                                          │
│       ▼                                                          │
│   工具执行 ◄────── 使用沙箱执行命令                                │
│       │                                                          │
│       ▼                                                          │
│   Agent 结束 ──→ after_agent() ──→ release() ──→ 释放沙箱         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  lazy_init=False                                                 │
│  ───────────────                                                 │
│                                                                  │
│   Agent 调用 ──→ before_agent() ──→ acquire() ──→ 立即获取沙箱    │
│       │                                                          │
│       ▼                                                          │
│   工具调用 ──→ 直接使用已有沙箱                                   │
│       │                                                          │
│       ▼                                                          │
│   Agent 结束 ──→ after_agent() ──→ release() ──→ 释放沙箱         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```text

关键点：
1. **线程级别复用** — 同一线程内的多次工具调用共享同一个沙箱
2. **自动释放** — Agent 结束时自动调用 `release()`
3. **状态持久化** — 沙箱状态保存在 `ThreadState` 中


## 8.6 安全检查机制

### 8.6.1 SandboxSecurity 类

```python
# packages/harness/deerflow/sandbox/security.py

"""Security helpers for sandbox capability gating."""

from deerflow.config import get_app_config

_LOCAL_SANDBOX_PROVIDER_MARKERS = (
    "deerflow.sandbox.local:LocalSandboxProvider",
    "deerflow.sandbox.local.local_sandbox_provider:LocalSandboxProvider",
)

LOCAL_HOST_BASH_DISABLED_MESSAGE = (
    "Host bash execution is disabled for LocalSandboxProvider because it is not a secure "
    "sandbox boundary. Switch to AioSandboxProvider for isolated bash access, or set "
    "sandbox.allow_host_bash: true only in a fully trusted local environment."
)

LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE = (
    "Bash subagent is disabled for LocalSandboxProvider because host bash execution is not "
    "a secure sandbox boundary. Switch to AioSandboxProvider for isolated bash access, or "
    "set sandbox.allow_host_bash: true only in a fully trusted local environment."
)