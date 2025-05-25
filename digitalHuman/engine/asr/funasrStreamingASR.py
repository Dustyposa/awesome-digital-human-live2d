# -*- coding: utf-8 -*-
# '''
# @File    :   funasrStreamingASR.py
# @Author  :   一力辉 (Modified by AI Assistant)
# '''

import numpy as np
from yacs.config import CfgNode as CN
from funasr_onnx.paraformer_online_bin import Paraformer
# from modelscope import snapshot_download # Potentially needed if model_dir is not local
from pathlib import Path

from ..builder import ASREngines # Added import
from digitalHuman.engine.engineBase import BaseEngine
from digitalHuman.utils import AudioMessage, TextMessage, logger # Assuming logger is available

@ASREngines.register("FunasrStreamingASR") # Added decorator
class FunasrStreamingASR(BaseEngine):
    # Funasr Streaming Automatic Speech Recognition Engine.
    # Uses FunASR ONNX Paraformer model for streaming speech recognition.
    def __init__(self, config: CN):
        super().__init__(config)
        # 根据配置初始化Paraformer模型
        model_dir_path = self.cfg.MODEL_DIR
        # 实际项目中, 此处可能需要更鲁棒的路径处理或自动下载逻辑
        # For example:
        # if not Path(model_dir_path).is_dir():
        #     logger.info(f"[FunasrStreamingASR] Model directory {model_dir_path} not found, attempting download...")
        #     model_dir_path = snapshot_download(self.cfg.MODEL_DIR, cache_dir="./models") # Define a cache_dir
        # else:
        #     logger.info(f"[FunasrStreamingASR] Using local model directory: {model_dir_path}")

        logger.info(f"[FunasrStreamingASR] Initializing Paraformer model from: {model_dir_path}")
        logger.info(f"[FunasrStreamingASR] Chunk size: {self.cfg.CHUNK_SIZE}")
        
        self.model = Paraformer(
            model_dir_path, 
            batch_size=self.cfg.BATCH_SIZE, 
            quantize=self.cfg.QUANTIZE, 
            chunk_size=self.cfg.CHUNK_SIZE, 
            intra_op_num_threads=self.cfg.INTRA_OP_NUM_THREADS
        )
        # 流式识别过程中的缓存, 每个识别会话应有独立的cache
        self.reset_stream() # Initialize cache for the instance
        
        # Paraformer的step是基于其内部处理单元(frame)定义的
        # 通常一个frame是10ms, 一个chunk包含6个frame (60ms)
        # step = chunk_size[1] * num_frames_per_chunk * frame_shift_ms * sample_rate / 1000
        # 根据funasr example: chunk_size[1] * 960 (where 960 samples = 60ms * 16kHz)
        self.step_samples = self.cfg.CHUNK_SIZE[1] * 960 
        logger.info(f"[FunasrStreamingASR] Model initialized. Step size for run method: {self.step_samples} samples.")

    def checkKeys(self) -> list[str]:
        # 定义此引擎配置必须包含的键
        return ["MODEL_DIR", "CHUNK_SIZE", "BATCH_SIZE", "QUANTIZE", "INTRA_OP_NUM_THREADS"]

    def _process_audio_chunk(self, audio_chunk: np.ndarray, param_dict: dict, is_final: bool) -> str:
        # 处理单个音频数据块 (numpy array).
        # :param audio_chunk: numpy array, float32, 语音数据块
        # :param param_dict: dict, 当前流的缓存字典
        # :param is_final: bool, 是否为最后一个数据块
        # :return: str, 识别结果文本
        param_dict['is_final'] = is_final
        try:
            rec_result = self.model(audio_in=audio_chunk, param_dict=param_dict)
            if rec_result and len(rec_result) > 0 and "preds" in rec_result[0]:
                return rec_result[0]["preds"][0]
        except Exception as e:
            logger.error(f"[FunasrStreamingASR] Error processing chunk: {e}")
        return ""

    async def run(self, input: AudioMessage, **kwargs) -> TextMessage:
        # 处理完整的AudioMessage (模拟流式处理整个音频).
        # 此方法会为当前调用重置并使用引擎内置的param_dict_cache.
        # 不适合并发调用处理多个完整音频流, 为此应使用 process_chunk 并外部管理 param_dict.
        logger.info(f"[FunasrStreamingASR] Received full audio for processing. Length: {len(input.data)} bytes, Sample Rate: {input.sampleRate}")
        
        if input.sampleRate != 16000:
            logger.warning(f"[FunasrStreamingASR] Sample rate is {input.sampleRate}, expected 16000. Audio quality might be affected. No resampling implemented.")

        try:
            speech_s16 = np.frombuffer(input.data, dtype=np.int16)
            speech_f32 = speech_s16.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"[FunasrStreamingASR] Could not convert audio data: {e}. Ensure input.data is raw 16-bit PCM mono audio bytes.")
            return TextMessage(data="")

        speech_length = speech_f32.shape[0]
        sample_offset = 0
        final_result_text = ""
        
        # 为本次run调用重置流状态 (使用实例缓存)
        current_run_param_dict = self.reset_stream() # Resets self.param_dict_cache and returns it

        logger.info(f"[FunasrStreamingASR] Starting simulated streaming for full audio. Total samples: {speech_length}")

        idx = 0
        while sample_offset < speech_length:
            end_offset = min(sample_offset + self.step_samples, speech_length)
            chunk = speech_f32[sample_offset:end_offset]
            is_final_chunk = (end_offset >= speech_length)
            
            recognized_text = self._process_audio_chunk(chunk, current_run_param_dict, is_final_chunk)
            if recognized_text:
                final_result_text += recognized_text
            
            sample_offset = end_offset
            idx +=1

        logger.info(f"[FunasrStreamingASR] Simulated streaming for full audio finished. Result: {final_result_text}")
        return TextMessage(data=final_result_text)

    def process_chunk(self, audio_chunk_bytes: bytes, param_dict: dict, is_final: bool, sample_rate: int = 16000) -> str:
        # 处理来自外部的单个音频数据块 (bytes). 主要供WebSocket服务器调用.
        # 调用方需要创建并传入 param_dict 以维持特定流式会话的状态.
        # :param audio_chunk_bytes: bytes, 原始音频数据块 (期望是16-bit PCM mono)
        # :param param_dict: dict, 当前流的缓存字典. 调用方负责创建和维护.
        # :param is_final: bool, 是否为最后一个数据块
        # :param sample_rate: int, 音频采样率
        # :return: str, 识别结果文本
        if sample_rate != 16000:
            logger.warning(f"[FunasrStreamingASR] Chunk sample rate is {sample_rate}, expected 16000. No resampling.")
            return ""

        try:
            audio_chunk_s16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
            audio_chunk_f32 = audio_chunk_s16.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"[FunasrStreamingASR] Error converting chunk bytes: {e}")
            return ""
        
        return self._process_audio_chunk(audio_chunk_f32, param_dict, is_final)

    def reset_stream(self) -> dict:
        # 重置/初始化引擎实例的流状态, 并返回一个新的缓存字典.
        # 在新的独立语音流开始 (且使用引擎实例自身缓存) 时调用.
        # :return: dict, 新的空缓存字典 (self.param_dict_cache)
        logger.info("[FunasrStreamingASR] Resetting instance stream state and cache.")
        self.param_dict_cache = {'cache': dict()} 
        return self.param_dict_cache

    def get_param_dict(self) -> dict:
        # 获取一个全新的独立的参数字典, 用于管理单个流式会话的缓存.
        # 推荐WebSocket服务为每个连接使用此方法获取独立的cache.
        # :return: dict, 新的空缓存字典
        logger.info("[FunasrStreamingASR] Providing new, independent cache for a stream.")
        return {'cache': dict()}

    def release(self):
        # 释放资源
        logger.info("[FunasrStreamingASR] Releasing resources.")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'param_dict_cache'): 
            self.param_dict_cache.clear()
