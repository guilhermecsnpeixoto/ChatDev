"""Register built-in agent providers."""

from runtime.node.agent.providers.base import ProviderRegistry

from runtime.node.agent.providers.iaedu_provider import IAeduProvider
from runtime.node.agent.providers.openai_provider import OpenAIProvider
from runtime.node.agent.providers.openrouter_provider import OpenRouterProvider

ProviderRegistry.register(
    "openai",
    OpenAIProvider,
    label="OpenAI",
    summary="OpenAI models via the official OpenAI SDK (responses API)",
)

ProviderRegistry.register(
    "openrouter",
    OpenRouterProvider,
    label="OpenRouter",
    summary="OpenRouter models via OpenAI-compatible API",
)

try:
    from runtime.node.agent.providers.gemini_provider import GeminiProvider
except ImportError:
    GeminiProvider = None

if GeminiProvider is not None:
    ProviderRegistry.register(
        "gemini",
        GeminiProvider,
        label="Google Gemini",
        summary="Google Gemini models via google-genai",
    )
else:
    print("Gemini provider not registered: google-genai library not found.")

ProviderRegistry.register(
    "iaedu",
    IAeduProvider,
    label="IAedu",
    summary="IAedu agent-chat endpoint via multipart/form-data requests",
)
