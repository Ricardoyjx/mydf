"""模型工厂：根据配置动态加载并创建 LLM 模型实例。"""

import importlib

from langchain.chat_models import BaseChatModel

from my_df.config.app_config import AppConfig


def create_chat_model(
    name: str | None = None,
    thinking_enable: bool = True,
    *,
    app_config: AppConfig,
    attach_tracing: bool = True,
    **kwargs,
) -> BaseChatModel:
    """根据应用配置动态创建聊天模型实例。

    参数：
        name:            模型配置名称。为 None 时使用配置列表中第一个模型。
        thinking_enable: 是否启用思考模式。
        app_config:      应用配置，包含模型列表。
        attach_tracing:  是否附加 LangSmith 追踪回调。

    返回：
        BaseChatModel 实例。

    抛出：
        ValueError: 配置中未定义任何模型。
    """
    models = app_config.models
    if not models:
        raise ValueError("未配置任何模型")

    # 按名称查找，未指定则取第一个
    config = (
        next((m for m in models if m.name == name), models[0]) if name else models[0]
    )

    # 动态导入模型类
    module_path, class_name = config.use.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    # 根据 thinking_enable 选择额外的传参
    extra_kwargs = (
        (config.when_thinking_enabled or {})
        if thinking_enable
        else (config.when_thinking_disabled or {})
    )

    return model_cls(model=config.model, **extra_kwargs, **kwargs)
