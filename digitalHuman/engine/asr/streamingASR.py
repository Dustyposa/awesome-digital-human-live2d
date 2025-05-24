import asyncio
import logging
import numpy as np
# import torch # Moved to NormalModelStrategy or if used by both
# import soundfile # If used, should be in specific strategy or remain if general
from yacs.config import CfgNode as CN
import abc # Added for ModelStrategy
import os # Added for ONNXModelStrategy path checking

from digitalHuman.engine.engineBase import BaseEngine
from ..builder import ASREngines

# Logger setup
logger = logging.getLogger(__name__)

# Suppress verbose ModelScope logging (can be kept here or moved if only one strategy uses it)
try:
    from modelscope.utils.logger import get_logger as ms_get_logger # Renamed to avoid conflict
    ms_logger = ms_get_logger(log_level=logging.CRITICAL)
    ms_logger.setLevel(logging.CRITICAL)
except Exception as e:
    logger.warning(f"Could not suppress ModelScope logger: {e}")


class ModelStrategy(abc.ABC):
    def __init__(self, engine_config: CN, specific_model_config: CN):
        self.engine_config = engine_config # Overall engine config
        self.model_config = specific_model_config # e.g., CHUNK_CONFIG.NORMAL_MODEL

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def get_initial_stream_state(self) -> dict:
        pass

    @abc.abstractmethod
    async def process_segment(self, model, audio_segment_np, stream_state: dict, is_final_engine_flag: bool) -> tuple[list[dict], dict]:
        pass

    @abc.abstractmethod
    def _normalize_result(self, raw_asr_output, is_final_segment_flag: bool) -> list[dict]:
        pass
    
    def get_model_stride_samples(self) -> int:
        # Assumes STRIDE_SIZE_MS is in self.model_config and sample rate is 16kHz.
        # Audio is 16-bit PCM, so 1 sample = 2 bytes. Stride is in samples.
        try:
            return self.model_config.STRIDE_SIZE_MS * 16 # 16 samples per ms for 16kHz
        except AttributeError as e:
            logger.error(f"STRIDE_SIZE_MS not found in model_config: {self.model_config}. Error: {e}")
            raise ValueError(f"STRIDE_SIZE_MS not configured for the model strategy") from e


class NormalModelStrategy(ModelStrategy):
    def __init__(self, engine_config: CN, specific_model_config: CN):
        super().__init__(engine_config, specific_model_config)
        # Specific imports for NormalModel (ModelScope)
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipeline = pipeline
        self.Tasks = Tasks
        logger.info("NormalModelStrategy initialized.")

    def load_model(self):
        model_id = self.engine_config.MODEL_PATH_NORMAL
        model_revision = self.engine_config.MODEL_REVISION_NORMAL
        logger.info(f"Loading normal model: {model_id} with revision: {model_revision}")
        try:
            model = self.pipeline(
                task=self.Tasks.auto_speech_recognition,
                model=model_id,
                model_revision=model_revision
            )
            logger.info(f"Successfully loaded normal model: {model_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load normal model ({model_id}): {e}", exc_info=True)
            raise

    def get_initial_stream_state(self) -> dict:
        return {'cache': {}} # ModelScope pipeline often uses a 'cache' dict

    async def process_segment(self, model, audio_segment_np, stream_state: dict, is_final_engine_flag: bool) -> tuple[list[dict], dict]:
        current_cache = stream_state.get('cache', {})
        model_call_kwargs = self.model_config.CHUNK_KWARGS.to_dict() if hasattr(self.model_config, 'CHUNK_KWARGS') else {}
        
        logger.debug(f"NormalModelStrategy processing segment. is_final_engine_flag: {is_final_engine_flag}, kwargs: {model_call_kwargs}")

        try:
            raw_res = await asyncio.to_thread(
                model,
                audio_in=audio_segment_np,
                # cache=current_cache, # Pass cache if model uses it this way explicitly; some pipelines handle it via instance state
                is_final=is_final_engine_flag,
                **model_call_kwargs
            )
        except Exception as e:
            logger.error(f"Error during NormalModel ASR processing: {e}", exc_info=True)
            return [], stream_state # Return empty results and original state on error

        # ModelScope's pipeline for streaming ASR often updates the cache internally or returns it.
        # If it's returned in `raw_res` (e.g. `raw_res = {'text': '...', 'cache': new_cache}`):
        updated_cache = raw_res.get('cache', current_cache) if isinstance(raw_res, dict) else current_cache
        # If the pipeline object `model` itself maintains the cache state across calls for a given "session"
        # (which is less common for a stateless `process_segment` call unless `model` is stream-specific),
        # then cache update might not be needed here or handled differently.
        # For now, assuming cache might be returned or updated if model is stateful.
        # If model is stateless per call, cache handling might be simpler or managed by the pipeline internally.
        
        # The prompt states "Updates and returns the cache".
        # Let's assume the pipeline might return a new cache object or relevant state information.
        # If the model mutates `current_cache` directly, `updated_cache` will reflect that if it's a dict.
        # A common pattern is `res_dict = model(...); new_cache = res_dict.get('cache')`.
        # Let's assume the pipeline instance itself might hold some state if not passed via cache dict.
        # For this refactor, we'll stick to explicit cache passing if the pipeline supports it.
        # The original code did: `if isinstance(raw_res, dict) and 'cache' in raw_res: self._active_streams[stream_id]['cache'] = raw_res['cache']`
        
        normalized_results = self._normalize_result(raw_res, is_final_engine_flag)
        
        # Update the stream state with the new cache
        stream_state['cache'] = updated_cache 
        return normalized_results, stream_state

    def _normalize_result(self, raw_asr_output, is_final_segment_flag: bool) -> list[dict]:
        results = []
        text = ""
        actual_is_final = is_final_segment_flag

        try:
            # ModelScope ASR pipeline typically returns a list of dictionaries or a single dict
            # e.g. [{'text': 'recognized speech', 'timestamp': [[...]], 'is_final': True/False (if supported)}]
            # or {'text': '...', 'is_final': ...}
            # The original code example used `res[0]["value"]` or `res[0]["text"]`.
            
            output_to_parse = None
            if isinstance(raw_asr_output, list) and len(raw_asr_output) > 0:
                output_to_parse = raw_asr_output[0] # Take the first element if it's a list
            elif isinstance(raw_asr_output, dict):
                output_to_parse = raw_asr_output # Use the dict directly
            
            if output_to_parse and isinstance(output_to_parse, dict):
                if "text" in output_to_parse:
                    text = output_to_parse["text"]
                elif "value" in output_to_parse: # Fallback for some model types
                    text = output_to_parse["value"]
                
                # Check if the model itself provides an 'is_final' flag for the segment
                if "is_final" in output_to_parse:
                    actual_is_final = output_to_parse["is_final"]
            else:
                logger.warning(f"Unexpected raw ASR output format (normal): {raw_asr_output}")

            # If the recognized text is empty, it's often not a meaningful partial result.
            # However, an empty string can be a valid final result (e.g. silence).
            if text or actual_is_final: # Send if text exists OR it's a final (potentially empty) segment
                results.append({"text": text, "is_partial": not actual_is_final})
        
        except Exception as e:
            logger.error(f"Error normalizing NormalModel result: {raw_asr_output}, error: {e}", exc_info=True)
            # Return empty list on error to avoid breaking the flow
            return []
            
        return results


class ONNXModelStrategy(ModelStrategy):
    def __init__(self, engine_config: CN, specific_model_config: CN):
        super().__init__(engine_config, specific_model_config)
        # Specific imports for ONNXModel (FunASR)
        from funasr_onnx.paraformer_online_bin import Paraformer
        from modelscope.utils.hub import snapshot_download # For downloading model if path is ID
        self.Paraformer = Paraformer
        self.snapshot_download = snapshot_download
        logger.info("ONNXModelStrategy initialized.")

    def load_model(self):
        model_path = self.engine_config.MODEL_PATH_ONNX
        
        # Check if model_path is a directory; if not, assume it's a model ID for snapshot_download
        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            logger.info(f"ONNX model path {model_path} not found locally, attempting download using snapshot_download.")
            try:
                # Use MODEL_REVISION_ONNX if provided in config, else None for default revision
                revision = self.engine_config.get("MODEL_REVISION_ONNX", None)
                model_path = self.snapshot_download(model_path, revision=revision)
                logger.info(f"ONNX model downloaded/found at: {model_path}")
            except Exception as e:
                logger.error(f"Failed to download ONNX model ({self.engine_config.MODEL_PATH_ONNX}): {e}", exc_info=True)
                raise

        onnx_constructor_args = self.model_config.CONSTRUCTOR_ARGS.to_dict()
        logger.info(f"Loading ONNX model from path: {model_path} with args: {onnx_constructor_args}")
        
        try:
            model = self.Paraformer(
                model_dir=model_path,
                **onnx_constructor_args
            )
            logger.info(f"Successfully loaded ONNX model from: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}", exc_info=True)
            raise

    def get_initial_stream_state(self) -> dict:
        # Paraformer's documentation suggests `param_dict` holds cache and other states.
        return {'param_dict': {'cache': {}}} 

    async def process_segment(self, model, audio_segment_np, stream_state: dict, is_final_engine_flag: bool) -> tuple[list[dict], dict]:
        param_dict = stream_state.get('param_dict', {'cache': {}}) # Get or initialize param_dict
        param_dict['is_final'] = is_final_engine_flag # Set final flag for this segment

        logger.debug(f"ONNXModelStrategy processing segment. is_final_engine_flag: {is_final_engine_flag}")
        
        try:
            # FunASR Paraformer expects audio_in as np.ndarray and param_dict
            # It updates param_dict['cache'] internally.
            raw_res = await asyncio.to_thread(
                model,
                audio_in=audio_segment_np,
                param_dict=param_dict
            )
        except Exception as e:
            logger.error(f"Error during ONNXModel ASR processing: {e}", exc_info=True)
            return [], stream_state # Return empty results and original state on error

        normalized_results = self._normalize_result(raw_res, is_final_engine_flag)
        
        # param_dict is mutated by the model call, so stream_state already reflects the update.
        return normalized_results, stream_state


    def _normalize_result(self, raw_asr_output, is_final_segment_flag: bool) -> list[dict]:
        results = []
        text = ""
        actual_is_final = is_final_segment_flag

        try:
            # FunASR ONNX Paraformer typically returns a list of dictionaries for streaming:
            # e.g., [{'text': ' recognized text segment.', 'timestamp': [[...]], 'is_final': False}]
            # Older funasr might have used `raw_res_list[0]["preds"][0]`.
            # Newer funasr_onnx.paraformer_online_bin.Paraformer returns list of dicts.
            if isinstance(raw_asr_output, list) and len(raw_asr_output) > 0:
                # Iterate if model can return multiple segments (though usually one for streaming chunk)
                for res_item in raw_asr_output:
                    current_text = ""
                    current_actual_is_final = is_final_segment_flag # Default to segment finality

                    if isinstance(res_item, dict) and "text" in res_item:
                        current_text = res_item["text"]
                        if "is_final" in res_item: # Check if the model provides its own final flag per item
                            current_actual_is_final = res_item["is_final"]
                    elif isinstance(res_item, dict) and "preds" in res_item and isinstance(res_item["preds"], list) and len(res_item["preds"]) > 0: # Older format
                        current_text = res_item["preds"][0]
                    else:
                        logger.warning(f"Unexpected raw ASR output format (onnx list item): {res_item}")

                    if current_text or current_actual_is_final:
                         results.append({"text": current_text, "is_partial": not current_actual_is_final})
            elif raw_asr_output: # If not a list, but not empty (e.g. single dict, though less common for FunASR list output)
                 logger.warning(f"Unexpected raw ASR output format (onnx, expected list): {raw_asr_output}")

        except Exception as e:
            logger.error(f"Error normalizing ONNXModel result: {raw_asr_output}, error: {e}", exc_info=True)
            return []
        
        return results


@ASREngines.register("StreamingASR")
class StreamingASR(BaseEngine):
    def __init__(self, config: CN):
        super().__init__(config)
        self.cfg = config # Overall config node for the engine
        self.model_handler: ModelStrategy = None
        self.model = None # Loaded model instance
        self.model_stride_samples = 0

        self._active_streams = {}
        self.internal_buffer = {}

        model_type = self.cfg.ENGINE.MODEL_TYPE # From main config, e.g., "normal" or "onnx"
        chunk_config = self.cfg.ENGINE.CHUNK_CONFIG # Specific chunk config root

        logger.info(f"Initializing StreamingASR engine with model type: {model_type}")

        try:
            if model_type == "normal":
                self.model_handler = NormalModelStrategy(self.cfg.ENGINE, chunk_config.NORMAL_MODEL)
            elif model_type == "onnx":
                self.model_handler = ONNXModelStrategy(self.cfg.ENGINE, chunk_config.ONNX_MODEL)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.model = self.model_handler.load_model()
            self.model_stride_samples = self.model_handler.get_model_stride_samples()
            logger.info(f"Model stride set to {self.model_stride_samples} samples ({self.model_stride_samples/16}ms)")

        except Exception as e:
            logger.error(f"Failed to initialize StreamingASR or load model: {e}", exc_info=True)
            raise

        self.setup()

    def checkKeys(self) -> list[str]:
        # These keys are expected under self.cfg.ENGINE
        # Specific model/chunk keys are implicitly checked by strategy instantiation or usage.
        required_keys = [
            "MODEL_TYPE",
            "CHUNK_CONFIG",
            # For NORMAL_MODEL (if type is normal)
            "MODEL_PATH_NORMAL", 
            "MODEL_REVISION_NORMAL",
            "CHUNK_CONFIG.NORMAL_MODEL.STRIDE_SIZE_MS",
            # "CHUNK_CONFIG.NORMAL_MODEL.CHUNK_KWARGS..." # Checked by strategy if needed
            # For ONNX_MODEL (if type is onnx)
            "MODEL_PATH_ONNX",
            # "MODEL_REVISION_ONNX", # Optional, handled by ONNX strategy
            "CHUNK_CONFIG.ONNX_MODEL.STRIDE_SIZE_MS",
            "CHUNK_CONFIG.ONNX_MODEL.CONSTRUCTOR_ARGS.BATCH_SIZE", # Example, actual check in strategy
        ]
        logger.debug(f"Basic expected config keys for StreamingASR: {required_keys}")
        # A more robust check would adapt to MODEL_TYPE.
        # For now, this lists common ones. Strategies handle their own specifics.
        return required_keys 

    def setup(self):
        logger.info("StreamingASR engine setup complete with chosen strategy.")

    def release(self):
        logger.info("Releasing StreamingASR engine resources.")
        if self.model is not None:
            # If model has a specific release method, call it here.
            # For ModelScope pipelines or FunASR objects, Python's GC is usually sufficient.
            del self.model
            self.model = None
        if self.model_handler is not None:
            del self.model_handler 
            self.model_handler = None
        self._active_streams = {}
        self.internal_buffer = {}
        logger.info("StreamingASR engine resources released.")

    async def init_stream(self, stream_id: str) -> None:
        logger.info(f"Initializing stream: {stream_id}")
        if stream_id in self._active_streams:
            logger.warning(f"Stream {stream_id} already initialized. Re-initializing.")
        
        if not self.model_handler:
            logger.error("Model handler not initialized. Cannot init stream.")
            raise RuntimeError("StreamingASR not properly initialized with a model strategy.")

        self._active_streams[stream_id] = self.model_handler.get_initial_stream_state()
        self.internal_buffer[stream_id] = bytearray()
        logger.info(f"Stream {stream_id} initialized successfully with state: {self._active_streams[stream_id]}.")


    async def process_chunk(self, stream_id: str, audio_chunk_data: bytes, is_final_client_chunk: bool) -> list[dict]:
        if stream_id not in self._active_streams:
            logger.error(f"Stream {stream_id} not initialized.")
            raise ValueError(f"Stream {stream_id} not initialized.")
        
        if not self.model_handler or not self.model:
            logger.error(f"Model or model_handler not available for stream {stream_id}.")
            raise RuntimeError("StreamingASR model/handler not ready.")

        self.internal_buffer[stream_id].extend(audio_chunk_data)
        current_buffer_samples = len(self.internal_buffer[stream_id]) // 2 # 2 bytes per sample
        
        processed_results = []

        while True:
            can_process_stride = current_buffer_samples >= self.model_stride_samples
            # Process if it's the final client chunk and there's *any* data left, even if less than a full stride.
            is_final_call_due_to_client_end = is_final_client_chunk and not can_process_stride and current_buffer_samples > 0

            if can_process_stride or is_final_call_due_to_client_end:
                samples_to_process = self.model_stride_samples if can_process_stride else current_buffer_samples
                
                if samples_to_process == 0: # Should not happen if logic is correct, but as a safeguard
                    # Exception: ONNX might need a final empty call if is_final_client_chunk is true
                    # This is usually handled by end_stream sending an empty final chunk if needed.
                    # Here, if samples_to_process is 0, we usually break.
                    # The ONNX strategy's process_segment should handle empty audio_segment_np if that's valid for it.
                    break

                audio_segment_bytes = self.internal_buffer[stream_id][:samples_to_process * 2]
                speech_segment_np = np.frombuffer(audio_segment_bytes, dtype=np.int16)

                self.internal_buffer[stream_id] = self.internal_buffer[stream_id][samples_to_process * 2:]
                current_buffer_samples = len(self.internal_buffer[stream_id]) // 2

                # This is True if this is the last piece of audio being sent to the model for this stream.
                is_final_engine_flag = is_final_client_chunk and current_buffer_samples == 0
                
                logger.debug(f"Processing chunk for stream {stream_id}: {samples_to_process} samples. IsFinalClient: {is_final_client_chunk}. IsFinalEngineCall: {is_final_engine_flag}")

                stream_specific_state = self._active_streams[stream_id]
                
                try:
                    normalized_segment_results, updated_stream_state = await self.model_handler.process_segment(
                        self.model,
                        speech_segment_np,
                        stream_specific_state,
                        is_final_engine_flag
                    )
                    self._active_streams[stream_id] = updated_stream_state
                    
                    # Filter out None or empty results unless it's a meaningful final empty result
                    for res in normalized_segment_results:
                        if res and (res.get("text") or not res.get("is_partial")): # Text exists or it's a final (potentially empty)
                            processed_results.append(res)
                            logger.debug(f"Normalized result for stream {stream_id}: {res}")
                        elif not res and is_final_engine_flag: # Strategy returned None/empty on final, make it a standard empty final
                             processed_results.append({"text": "", "is_partial": False})


                except Exception as e:
                    logger.error(f"Error during ASR processing segment for stream {stream_id}: {e}", exc_info=True)
                    if is_final_engine_flag: # Don't get stuck in a loop on final segment
                        break
                    # For non-final segments, we might choose to continue trying with next data or break.
                    # Breaking here to be safe.
                    break 
            
            else: # Not enough data for a stride and not the final call scenario
                break
        
        return processed_results

    async def end_stream(self, stream_id: str) -> list[dict]:
        logger.info(f"Ending stream: {stream_id}")
        final_results = []

        if not self.model_handler or not self.model:
            logger.warning(f"Model handler or model not available during end_stream for {stream_id}. Skipping processing.")
        # Process any remaining audio in the buffer as the absolute final chunk.
        elif stream_id in self.internal_buffer and len(self.internal_buffer[stream_id]) > 0:
            logger.info(f"Processing remaining {len(self.internal_buffer[stream_id])} bytes for stream {stream_id} before closing.")
            # Call process_chunk with is_final_client_chunk=True and empty additional data (b'').
            # This tells process_chunk to process whatever is left and signal finality.
            remaining_results = await self.process_chunk(stream_id, b'', is_final_client_chunk=True)
            if remaining_results:
                final_results.extend(remaining_results)
        # If buffer is empty but stream was active, ensure one last call to model if strategy requires (e.g. ONNX with empty final frame)
        elif stream_id in self._active_streams:
            logger.info(f"No remaining audio for stream {stream_id}, ensuring final signal to model if necessary.")
            stream_specific_state = self._active_streams[stream_id]
            # Check if this stream_specific_state indicates it's already finalized (e.g. ONNX param_dict['is_final'] = True)
            # This logic is now more encapsulated in the strategy's process_segment.
            # We send an empty audio segment with is_final_engine_flag=True. The strategy decides what to do.
            empty_audio = np.array([], dtype=np.int16)
            try:
                normalized_empty_final_results, updated_stream_state = await self.model_handler.process_segment(
                    self.model,
                    empty_audio,
                    stream_specific_state,
                    is_final_engine_flag=True # Crucial: this is the final signal
                )
                self._active_streams[stream_id] = updated_stream_state
                for res in normalized_empty_final_results:
                    if res and (res.get("text") or not res.get("is_partial")):
                        final_results.append(res)
                    elif not res : # Strategy returned None/empty on final, make it a standard empty final
                        final_results.append({"text": "", "is_partial": False})

            except Exception as e:
                logger.error(f"Error sending final empty chunk for stream {stream_id}: {e}", exc_info=True)

        # Clean up
        if stream_id in self._active_streams:
            del self._active_streams[stream_id]
            logger.debug(f"Active stream state/cache deleted for {stream_id}")
        if stream_id in self.internal_buffer:
            del self.internal_buffer[stream_id]
            logger.debug(f"Internal buffer deleted for {stream_id}")

        logger.info(f"Stream {stream_id} ended. Final results count: {len(final_results)}")
        return final_results

```
