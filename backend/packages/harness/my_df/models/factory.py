import importlib

from langchain.chat_models import BaseChatModel

from backend.packages.harness.my_df.agents.config.app_config import AppConfig


def create_chat_model(
    name: str | None = None,
    thinking_enable: bool = True,
    *,
    app_config: AppConfig,
    attach_tracing: bool = True,
    **kwargs,
) -> BaseChatModel:

    models = app_config.models
    if not models:
        raise ValueError("No models found")

    config = (
        next((m for m in models if m.name == name), models[0]) if name else models[0]
    )

    module_path, class_name = config.use.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    extra_kwargs = (
        (config.when_thinking_enabled or {})
        if thinking_enable
        else (config.when_thinking_disabled or {})
    )

    return model_cls(model=config.model, **extra_kwargs, **kwargs)
