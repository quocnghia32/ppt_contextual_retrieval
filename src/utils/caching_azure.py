"""
Caching utilities for OpenAI models (Embeddings + LLMs).

Implements:
1. CacheBackedEmbeddings for embedding caching
2. LLM response caching
3. Semantic caching for similar queries
"""
from typing import Optional, List
from pathlib import Path
import hashlib
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore, InMemoryStore
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache, InMemoryCache
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from loguru import logger

from src.config import settings


class AzureOpenAICachingManager:
    """
    Manages caching for OpenAI models.

    Provides:
    - Embedding caching (CacheBackedEmbeddings)
    - LLM response caching (SQLite or In-Memory)
    - Cache invalidation and management
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_in_memory: bool = False,
        enable_llm_cache: bool = True
    ):
        """
        Initialize caching manager.

        Args:
            cache_dir: Directory for cache files (default: settings.cache_dir)
            use_in_memory: Use in-memory cache instead of disk
            enable_llm_cache: Enable LLM response caching
        """
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        self.use_in_memory = use_in_memory
        self.enable_llm_cache = enable_llm_cache

        # Create cache directories
        self.embeddings_cache_dir = self.cache_dir / "embeddings"
        self.llm_cache_dir = self.cache_dir / "llm_responses"

        if not use_in_memory:
            self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)
            self.llm_cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup LLM cache
        if enable_llm_cache:
            self._setup_llm_cache()

        logger.info(
            f"Caching manager initialized: "
            f"cache_dir={self.cache_dir}, in_memory={use_in_memory}"
        )

    def _setup_llm_cache(self):
        """Setup global LLM cache for LangChain."""
        if self.use_in_memory:
            set_llm_cache(InMemoryCache())
            logger.info("LLM caching: In-Memory")
        else:
            cache_path = self.llm_cache_dir / "llm_cache.db"
            set_llm_cache(SQLiteCache(database_path=str(cache_path)))
            logger.info(f"LLM caching: SQLite at {cache_path}")

    def create_cached_embeddings(
        self,
        model: str = None,
        namespace: Optional[str] = None
    ) -> CacheBackedEmbeddings:
        """
        Create cached embeddings instance.

        Args:
            model: OpenAI embedding model (default: from settings)
            namespace: Cache namespace (default: model name)

        Returns:
            CacheBackedEmbeddings instance
        """
        model = model or settings.embedding_model
        namespace = namespace or f"azure_embeddings_{settings.azure_openai_embedding_deployment.replace('-', '_')}"

        underlying_embeddings = AzureOpenAIEmbeddings(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version_embedding,
            azure_deployment=settings.azure_openai_embedding_deployment,
        )

        # Create cache store
        if self.use_in_memory:
            store = InMemoryStore()
        else:
            cache_path = self.embeddings_cache_dir / namespace
            cache_path.mkdir(exist_ok=True)
            store = LocalFileStore(str(cache_path))

        # Create cached embeddings
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=underlying_embeddings,
            document_embedding_cache=store,
            namespace=namespace
        )

        logger.info(f"Cached embeddings created: model={model}, namespace={namespace}")
        return cached_embeddings

    def create_cached_llm(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AzureChatOpenAI:
        """
        Create AzureChatOpenAI LLM with automatic caching.

        Args:
            model: AzureChatOpenAI model name
            temperature: Model temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional AzureChatOpenAI parameters

        Returns:
            AzureChatOpenAI instance with caching enabled
        """
        llm = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version_chat,
            azure_deployment=settings.azure_openai_chat_deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        logger.info(f"Cached LLM created: model={model}")
        return llm

    def clear_embedding_cache(self, namespace: Optional[str] = None):
        """
        Clear embedding cache.

        Args:
            namespace: Specific namespace to clear (None = clear all)
        """
        if self.use_in_memory:
            logger.warning("Cannot clear in-memory cache (will be cleared on restart)")
            return

        if namespace:
            cache_path = self.embeddings_cache_dir / namespace
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                logger.info(f"Cleared embedding cache: {namespace}")
        else:
            import shutil
            if self.embeddings_cache_dir.exists():
                shutil.rmtree(self.embeddings_cache_dir)
                self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all embedding caches")

    def clear_llm_cache(self):
        """Clear LLM response cache."""
        if self.use_in_memory:
            set_llm_cache(InMemoryCache())  # Reset
            logger.info("LLM cache cleared (in-memory)")
        else:
            cache_path = self.llm_cache_dir / "llm_cache.db"
            if cache_path.exists():
                cache_path.unlink()
                self._setup_llm_cache()  # Recreate
                logger.info("LLM cache cleared (SQLite)")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        stats = {
            "cache_dir": str(self.cache_dir),
            "in_memory": self.use_in_memory,
            "llm_cache_enabled": self.enable_llm_cache
        }

        if not self.use_in_memory:
            # Count embedding cache files
            embedding_namespaces = list(self.embeddings_cache_dir.iterdir())
            stats["embedding_namespaces"] = len(embedding_namespaces)

            total_size = 0
            for namespace_dir in embedding_namespaces:
                if namespace_dir.is_dir():
                    for file in namespace_dir.rglob("*"):
                        if file.is_file():
                            total_size += file.stat().st_size

            stats["embedding_cache_size_mb"] = round(total_size / (1024 * 1024), 2)

            # LLM cache size
            llm_cache_path = self.llm_cache_dir / "llm_cache.db"
            if llm_cache_path.exists():
                stats["llm_cache_size_mb"] = round(
                    llm_cache_path.stat().st_size / (1024 * 1024), 2
                )
            else:
                stats["llm_cache_size_mb"] = 0

        return stats

print("HELLO WORLD FROM AZURE")
# Global caching manager instance
caching_manager = AzureOpenAICachingManager(
    cache_dir=settings.cache_dir,
    use_in_memory=False,  # Use disk-based caching for persistence
    enable_llm_cache=True
)


# Convenience functions
def get_cached_embeddings_azure(model: str = None) -> CacheBackedEmbeddings:
    """
    Get cached embeddings instance.

    Args:
        model: ) -> AzureChatOpenAI embedding model

    Returns:
        CacheBackedEmbeddings
    """
    return caching_manager.create_cached_embeddings(model=model)


def get_cached_llm_azure(
    model: str = "gpt-4o",
    temperature: float = 0.0,
    **kwargs
) -> AzureChatOpenAI:
    """
    Get cached OpenAI LLM.

    Args:
        model: Model name (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.)
        temperature: Temperature
        **kwargs: Additional parameters

    Returns:
        AzureChatOpenAI with caching
    """
    return caching_manager.create_cached_llm(
        model=model,
        temperature=temperature,
        **kwargs
    )


def clear_all_caches():
    """Clear all caches (embeddings + LLM responses)."""
    caching_manager.clear_embedding_cache()
    caching_manager.clear_llm_cache()
    logger.info("All caches cleared")


# Prompt optimization helpers for OpenAI caching
class PromptOptimizer:
    """
    Helper to optimize prompts for OpenAI's automatic caching.

    OpenAI caches prompts > 1024 tokens with 128-token increments.
    """

    @staticmethod
    def structure_for_caching(
        static_instructions: str,
        static_examples: Optional[str] = None,
        dynamic_content: Optional[str] = None
    ) -> str:
        """
        Structure prompt to maximize cache hits.

        OpenAI best practice: Static content first, dynamic content last.

        Args:
            static_instructions: System instructions (cached)
            static_examples: Few-shot examples (cached)
            dynamic_content: User-specific content (not cached)

        Returns:
            Optimized prompt string
        """
        parts = []

        # Static parts first (will be cached)
        if static_instructions:
            parts.append(f"<instructions>\n{static_instructions}\n</instructions>")

        if static_examples:
            parts.append(f"<examples>\n{static_examples}\n</examples>")

        # Dynamic content last (not cached but benefits from cached prefix)
        if dynamic_content:
            parts.append(f"<input>\n{dynamic_content}\n</input>")

        return "\n\n".join(parts)

    @staticmethod
    def estimate_cache_savings(prompt: str) -> dict:
        """
        Estimate potential cache savings.

        Args:
            prompt: The prompt string

        Returns:
            Dict with savings estimates
        """
        # Rough token estimate (1 token ≈ 4 chars for English)
        estimated_tokens = len(prompt) / 4

        # OpenAI caches if > 1024 tokens
        cacheable = estimated_tokens > 1024

        if cacheable:
            # Cached prefix is longest 1024-token-aligned prefix
            cached_tokens = int(estimated_tokens / 128) * 128
            if cached_tokens < 1024:
                cached_tokens = 0

            # 50% discount on cached tokens
            savings_percent = (cached_tokens / estimated_tokens) * 50 if estimated_tokens > 0 else 0
        else:
            cached_tokens = 0
            savings_percent = 0

        return {
            "total_tokens": int(estimated_tokens),
            "cached_tokens": cached_tokens,
            "cacheable": cacheable,
            "estimated_savings_percent": round(savings_percent, 1),
            "recommendation": (
                "✅ Good for caching" if cacheable
                else "⚠️ Too short for automatic caching (needs >1024 tokens)"
            )
        }


# Global prompt optimizer
prompt_optimizer = PromptOptimizer()
