import logging
from abc import ABC
from collections.abc import Awaitable
from typing import Callable, Optional, Union

from urllib.parse import urljoin

import aiohttp
import tiktoken

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI, RateLimitError

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from typing_extensions import TypedDict

logger = logging.getLogger("scripts")


class EmbeddingBatch:
    """
    Represents a batch of text that is going to be embedded
    """

    def __init__(self,texts : list[str],token_length : int):
        self.texts = texts
        self.token_length = token_length


class ExtraArgs(TypedDict,total = False):
    dimensions : int


class OpenAIEmbeddings(ABC):
    """
    Contains common logic across both OpenAI and Azure OpenAI embedding services
    Can split source text into batches for more efficient embedding calls
    """

    SUPPORTED_BATCH_AOAI_MODEL = {
        "text-embedding-ada-002": {"token_limit": 8100, "max_batch_size": 16},
        "text-embedding-3-small": {"token_limit": 8100, "max_batch_size": 16},
        "text-embedding-3-large": {"token_limit": 8100, "max_batch_size": 16},
    }

    SUPPORTED_DIMENSIONS_MODEL = {
        "text-embedding-ada-002": False,
        "text-embedding-3-small": True,
        "text-embedding-3-large": True,
    }

    def __init__(self, open_ai_model_name: str, open_ai_dimensions: int, disable_batch: bool = False):
        self.open_ai_model_name = open_ai_model_name
        self.open_ai_dimensions = open_ai_dimensions
        self.disable_batch = disable_batch


    async def create_client(self) -> AsyncOpenAI:
        raise NotImplementedError
    
    def before_retry_sleep(self,retry_state):
         logger.info("Rate limited on the OpenAI embeddings API, sleeping before retrying...")


    def calculate_token_length(self,text : str):
        encoding = tiktoken.encoding_for_model(self.open_ai_model_name)

        return len(encoding.encode(text))
    
    def split_text_into_batches(self,texts : list[str]) -> list[EmbeddingBatch]:
        batch_info = OpenAIEmbeddings.SUPPORTED_BATCH_AOAI_MODEL.get(self.open_ai_model_name)
        if not batch_info:
            raise NotImplementedError( logger.info("Rate limited on the OpenAI embeddings API, sleeping before retrying..."))
        

        batch_token_limit = batch_info["token_limit"]
        batch_max_size = batch_info["max_batch_size"]
        batches : list[EmbeddingBatch] = []
        batch : list[str] = []
        batch_token_length = 0

        for text in texts:
            text_token_length = self.calculate_token_length(text)
            if batch_token_length + text_token_length >= batch_token_limit and len(batch) > 0:
                 batches.append(EmbeddingBatch(batch, batch_token_length))
                 batch = []
                 batch_token_length = 0


            batch.append(text)
            batch_token_length = batch_token_length + text_token_length

            if len(batch) ==  batch_max_size:
                batches.append(EmbeddingBatch(batch, batch_token_length))
                batch = []
                batch_token_length = 0


        if len(batch) > 0:
            batches.append(EmbeddingBatch(batch, batch_token_length))
        return batches
    

    async def create_embedding_batch(self,texts : list[str],dimensions_args : ExtraArgs) -> list[list[float]]:
        batches = self.split_text_into_batches(texts)
        embeddings = []
        client = await self.create_client()

        for batch in batches:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(RateLimitError),
                wait=wait_random_exponential(min=15, max=60),
                stop=stop_after_attempt(15),
                before_sleep=self.before_retry_sleep,
            ):
                with attempt:
                    emb_response = await client.embeddings.create(
                        model = self.open_ai_model_name,
                        input=batch.texts, 
                        **dimensions_args
                    )

                    embeddings.extend([data.embedding for data in emb_response.data])
                    logger.info(
                        "Computed embeddings in batch. Batch size: %d, Token count: %d",
                        len(batch.texts),
                        batch.token_length,
                    )


        return embeddings
    


    async def create_embedding_single(self,text : str,dimensions_args: ExtraArgs) -> list[float]:
         client = await self.create_client()
         async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=15, max=60),
            stop=stop_after_attempt(15),
            before_sleep=self.before_retry_sleep,
         ):
             with attempt:
                 emb_response = await client.embeddings.create(
                     model=self.open_ai_model_name, input=text, **dimensions_args
                 )
                 logger.info("Computed embedding for text section. Character count: %d", len(text))


         return emb_response.data[0].embedding
    

    async def create_embeddings(self,texts : list[str]) -> list[list[float]]:

        dimensions_args : ExtraArgs  = (
            {"dimensions": self.open_ai_dimensions}
            if OpenAIEmbeddings.SUPPORTED_DIMENSIONS_MODEL.get(self.open_ai_model_name)
            else {}
        )
        if not self.disable_batch and self.open_ai_model_name in OpenAIEmbeddings.SUPPORTED_BATCH_AOAI_MODEL:
            return await self.create_embedding_batch(texts,dimensions_args)
        

        return [await self.create_embedding_single(text,dimensions_args) for text in texts]
    
        
        
