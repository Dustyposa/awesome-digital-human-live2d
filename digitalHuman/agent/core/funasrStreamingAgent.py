# -*- coding: utf-8 -*-
# '''
# @File    :   funasrStreamingAgent.py
# @Author  :   AI Assistant
# '''

from yacs.config import CfgNode as CN
from typing import Union, AsyncGenerator, Optional

from digitalHuman.agent.agentBase import BaseAgent
from digitalHuman.utils import TextMessage, AudioMessage, logger
from digitalHuman.engine import EnginePool, EngineType # Required for getting the engine
from digitalHuman.engine.asr.funasrStreamingASR import FunasrStreamingASR # Specific engine type hint

class FunasrStreamingAgent(BaseAgent):
    # Funasr Streaming ASR Agent
    # This agent utilizes the FunasrStreamingASR engine to perform speech recognition.
    # It supports both a regular 'run' for full audio and a 'stream_run' for chunked audio processing via WebSocket.

    def __init__(self, config: CN):
        super().__init__(config)
        self.engine_pool = EnginePool() # Get an instance of EnginePool
        self.asr_engine: Optional[FunasrStreamingASR] = None
        self.setup_engine()

    def checkKeys(self) -> list[str]:
        # Keys required in the agent's configuration file
        return ["ENGINE_TYPE", "ENGINE_NAME"]

    def setup_engine(self):
        # 获取配置的ASR引擎实例
        engine_type_str = self.cfg.ENGINE_TYPE.upper()
        if engine_type_str == "ASR":
            engine_type = EngineType.ASR
        # Add other types like TTS, LLM if this agent were to support them
        else:
            logger.error(f"[{self.__class__.__name__}] Unsupported engine type: {self.cfg.ENGINE_TYPE}")
            raise ValueError(f"Unsupported engine type: {self.cfg.ENGINE_TYPE}")

        logger.info(f"[{self.__class__.__name__}] Initializing with engine: {self.cfg.ENGINE_NAME} of type {self.cfg.ENGINE_TYPE}")
        engine_instance = self.engine_pool.getEngine(engine_type, self.cfg.ENGINE_NAME)
        
        if not isinstance(engine_instance, FunasrStreamingASR):
            logger.error(f"[{self.__class__.__name__}] Engine {self.cfg.ENGINE_NAME} is not an instance of FunasrStreamingASR.")
            raise TypeError(f"Engine {self.cfg.ENGINE_NAME} is not an instance of FunasrStreamingASR.")
        self.asr_engine = engine_instance
        
        if self.asr_engine is None:
            logger.error(f"[{self.__class__.__name__}] Failed to get ASR engine: {self.cfg.ENGINE_NAME}")
            raise RuntimeError(f"Failed to get ASR engine: {self.cfg.ENGINE_NAME}")
        logger.info(f"[{self.__class__.__name__}] ASR Engine {self.cfg.ENGINE_NAME} initialized successfully.")

    async def run(
        self, 
        input: Union[TextMessage, AudioMessage], 
        streaming: bool, # This flag indicates if streaming mode is requested
        **kwargs
    ) -> Optional[TextMessage]:
        # 主要的运行逻辑
        if not self.asr_engine:
            logger.error(f"[{self.__class__.__name__}] ASR engine not initialized.")
            return TextMessage(data="Error: ASR engine not initialized.")

        if not isinstance(input, AudioMessage):
            logger.warning(f"[{self.__class__.__name__}] Received non-audio input, cannot process.")
            return TextMessage(data="Error: Input must be AudioMessage.")

        if streaming:
            # 流式处理的逻辑不直接在此处处理完整流程,
            # WebSocket服务器会调用asr_engine.process_chunk()
            # 此处返回一个提示或者初始握手(如果需要)
            # 或者, 如果这个run方法被WebSocket服务器的某个连接管理器调用，
            # 它可能只是返回一个确认，实际的流式数据通过另一个路径发送给asr_engine.process_chunk
            logger.info(f"[{self.__class__.__name__}] Streaming mode requested. Actual streaming is handled by WebSocket endpoint via process_chunk.")
            # For now, let's assume this method isn't directly called for established websocket streams.
            # If it IS the entry point, it would need to return an AsyncGenerator or similar.
            # Based on the issue, the websocket endpoint will directly use the engine.
            # This 'run' method with streaming=True might be for initiating a stream conversation_id if needed.
            return TextMessage(data="Streaming ASR session started. Send audio chunks via WebSocket.")
        else:
            # 非流式处理, 直接调用引擎的run方法处理整个音频
            logger.info(f"[{self.__class__.__name__}] Non-streaming mode. Processing full audio.")
            return await self.asr_engine.run(input, **kwargs)

    async def process_audio_chunk(self, audio_chunk_bytes: bytes, param_dict: dict, is_final: bool, sample_rate: int = 16000) -> str:
        # 暴露给WebSocket服务器调用的方法，直接调用引擎的chunk处理
        if not self.asr_engine:
            logger.error(f"[{self.__class__.__name__}] ASR engine not initialized for chunk processing.")
            return "" # Return empty string or raise error
        # logger.debug(f"[{self.__class__.__name__}] Processing audio chunk via engine. Is_final: {is_final}")
        return self.asr_engine.process_chunk(audio_chunk_bytes, param_dict, is_final, sample_rate)

    def get_engine_param_dict(self) -> dict:
        # 为WebSocket连接获取独立的参数字典
        if not self.asr_engine:
            logger.error(f"[{self.__class__.__name__}] ASR engine not initialized, cannot get param_dict.")
            raise RuntimeError("ASR Engine not initialized in Agent")
        return self.asr_engine.get_param_dict()
        
    def release(self):
        # 释放资源 (引擎的释放由EnginePool管理)
        logger.info(f"[{self.__class__.__name__}] Agent released. Engine itself is managed by EnginePool.")
        pass
