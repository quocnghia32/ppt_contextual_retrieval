"""
Configuration management for PPT Contextual Retrieval system.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")

    # Pinecone Configuration
    pinecone_environment: str = Field("us-east-1", env="PINECONE_ENVIRONMENT")
    pinecone_region: str = Field("us-east-1", env="PINECONE_REGION")
    pinecone_index_name: str = Field("ppt-contextual-retrieval", env="PINECONE_INDEX_NAME")

    # LangSmith Configuration (Optional)
    langchain_tracing_v2: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("ppt-contextual-retrieval", env="LANGCHAIN_PROJECT")

    # Rate Limiting
    max_requests_per_minute: int = Field(50, env="MAX_REQUESTS_PER_MINUTE")
    max_tokens_per_minute: int = Field(100000, env="MAX_TOKENS_PER_MINUTE")

    # Model Configuration
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")

    # Context generation: OpenAI or Anthropic
    context_generation_provider: str = Field("openai", env="CONTEXT_GENERATION_PROVIDER")  # "openai" or "anthropic"
    context_generation_model: str = Field("gpt-4o-mini", env="CONTEXT_GENERATION_MODEL")  # OpenAI: gpt-4o-mini, gpt-4o, gpt-3.5-turbo | Anthropic: claude-3-haiku

    # Answer generation: OpenAI or Anthropic
    answer_generation_provider: str = Field("openai", env="ANSWER_GENERATION_PROVIDER")  # "openai" or "anthropic"
    answer_generation_model: str = Field("gpt-4o", env="ANSWER_GENERATION_MODEL")  # OpenAI: gpt-4o, gpt-4-turbo, gpt-3.5-turbo | Anthropic: claude-3-sonnet

    # Vision model (OpenAI only)
    vision_model: str = Field("gpt-4o-mini", env="VISION_MODEL")  # gpt-4o-mini, gpt-4o, gpt-4-turbo

    # Caching configuration
    enable_embedding_cache: bool = Field(True, env="ENABLE_EMBEDDING_CACHE")
    enable_llm_cache: bool = Field(True, env="ENABLE_LLM_CACHE")
    cache_in_memory: bool = Field(False, env="CACHE_IN_MEMORY")

    # Chunking Configuration
    max_chunk_size: int = Field(400, env="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")

    # Retrieval Configuration
    top_k_retrieval: int = Field(20, env="TOP_K_RETRIEVAL")
    top_n_rerank: int = Field(5, env="TOP_N_RERANK")

    # Paths
    data_dir: str = Field("data", env="DATA_DIR")
    upload_dir: str = Field("data/uploads", env="UPLOAD_DIR")
    cache_dir: str = Field("data/cache", env="CACHE_DIR")
    extracted_images_dir: str = Field("data/extracted_images", env="EXTRACTED_IMAGES_DIR")

    # Logging
    verbose_logging: bool = Field(False, env="VERBOSE_LOGGING")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.extracted_images_dir, exist_ok=True)


# Global settings instance
settings = Settings()


# Set LangSmith environment variables if enabled
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
