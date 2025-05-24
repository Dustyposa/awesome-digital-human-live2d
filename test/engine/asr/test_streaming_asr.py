import pytest
import asyncio
from unittest.mock import MagicMock, patch, call, AsyncMock
import numpy as np
from yacs.config import CfgNode as CN

from digitalHuman.engine.asr.streamingASR import StreamingASR, NormalModelStrategy, ONNXModelStrategy

# Default configuration values
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_STRIDE_MS_NORMAL = 480
DEFAULT_STRIDE_MS_ONNX = 480
DEFAULT_SAMPLES_PER_CHUNK_NORMAL = int(DEFAULT_SAMPLE_RATE * DEFAULT_STRIDE_MS_NORMAL / 1000)
DEFAULT_SAMPLES_PER_CHUNK_ONNX = int(DEFAULT_SAMPLE_RATE * DEFAULT_STRIDE_MS_ONNX / 1000)

def get_mock_config(model_type="normal", stride_ms=None):
    cfg = CN()
    cfg.ENGINE = CN()
    cfg.ENGINE.MODEL_TYPE = model_type
    cfg.ENGINE.MODEL_PATH_NORMAL = "mock/normal_model_path"
    cfg.ENGINE.MODEL_REVISION_NORMAL = "v1.0"
    cfg.ENGINE.MODEL_PATH_ONNX = "mock/onnx_model_path" # Can be path or ID
    cfg.ENGINE.MODEL_REVISION_ONNX = "v1.1"

    cfg.ENGINE.CHUNK_CONFIG = CN()
    cfg.ENGINE.CHUNK_CONFIG.NORMAL_MODEL = CN()
    cfg.ENGINE.CHUNK_CONFIG.NORMAL_MODEL.STRIDE_SIZE_MS = stride_ms if stride_ms is not None else DEFAULT_STRIDE_MS_NORMAL
    cfg.ENGINE.CHUNK_CONFIG.NORMAL_MODEL.CHUNK_KWARGS = CN()
    cfg.ENGINE.CHUNK_CONFIG.NORMAL_MODEL.CHUNK_KWARGS.encoder_chunk_look_back = 4
    cfg.ENGINE.CHUNK_CONFIG.NORMAL_MODEL.CHUNK_KWARGS.decoder_chunk_look_back = 1

    cfg.ENGINE.CHUNK_CONFIG.ONNX_MODEL = CN()
    cfg.ENGINE.CHUNK_CONFIG.ONNX_MODEL.STRIDE_SIZE_MS = stride_ms if stride_ms is not None else DEFAULT_STRIDE_MS_ONNX
    cfg.ENGINE.CHUNK_CONFIG.ONNX_MODEL.CONSTRUCTOR_ARGS = CN()
    cfg.ENGINE.CHUNK_CONFIG.ONNX_MODEL.CONSTRUCTOR_ARGS.BATCH_SIZE = 1
    cfg.ENGINE.CHUNK_CONFIG.ONNX_MODEL.CONSTRUCTOR_ARGS.QUANTIZE = True
    cfg.ENGINE.CHUNK_CONFIG.ONNX_MODEL.CONSTRUCTOR_ARGS.CHUNK_SIZE = [5, 10, 5] # Example
    cfg.ENGINE.CHUNK_CONFIG.ONNX_MODEL.CONSTRUCTOR_ARGS.INTRA_OP_NUM_THREADS = 1
    return cfg

@pytest.mark.asyncio
class TestNormalModelStrategy:
    @patch('digitalHuman.engine.asr.streamingASR.NormalModelStrategy.pipeline') # Mocking the pipeline factory
    def test_load_model(self, mock_pipeline_factory):
        mock_pipeline_instance = MagicMock()
        mock_pipeline_factory.return_value = mock_pipeline_instance
        
        config = get_mock_config(model_type="normal")
        strategy = NormalModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.NORMAL_MODEL)
        model = strategy.load_model()

        mock_pipeline_factory.assert_called_once_with(
            task=strategy.Tasks.auto_speech_recognition, # Assuming Tasks is accessible or mocked
            model=config.ENGINE.MODEL_PATH_NORMAL,
            model_revision=config.ENGINE.MODEL_REVISION_NORMAL
        )
        assert model == mock_pipeline_instance

    def test_get_initial_stream_state(self):
        config = get_mock_config(model_type="normal")
        strategy = NormalModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.NORMAL_MODEL)
        initial_state = strategy.get_initial_stream_state()
        assert initial_state == {'cache': {}}

    async def test_process_segment(self):
        config = get_mock_config(model_type="normal")
        strategy = NormalModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.NORMAL_MODEL)
        
        mock_model = AsyncMock() # Mock the loaded pipeline model instance
        mock_model.return_value = {"text": "test output", "cache": {"new_key": "new_value"}} # Mock pipeline output
        
        audio_segment_np = np.random.randn(DEFAULT_SAMPLES_PER_CHUNK_NORMAL).astype(np.int16)
        initial_stream_state = {'cache': {'key': 'value'}}
        
        # Mock _normalize_result to isolate testing of process_segment logic
        strategy._normalize_result = MagicMock(return_value=[{"text": "normalized test output", "is_partial": False}])

        results, updated_state = await strategy.process_segment(mock_model, audio_segment_np, initial_stream_state, is_final_engine_flag=False)

        mock_model.assert_awaited_once()
        # Check args of the mocked model call. The first arg is 'self' if it's a method,
        # but here mock_model is the pipeline instance itself (a callable).
        # We need to ensure audio_in, is_final, and CHUNK_KWARGS are passed.
        # The actual call is made via asyncio.to_thread, so we check the args passed to the underlying callable.
        # The `mock_model` itself is the callable here.
        called_args, called_kwargs = mock_model.call_args
        assert np.array_equal(called_kwargs['audio_in'], audio_segment_np)
        assert called_kwargs['is_final'] == False
        assert called_kwargs['encoder_chunk_look_back'] == config.ENGINE.CHUNK_CONFIG.NORMAL_MODEL.CHUNK_KWARGS.encoder_chunk_look_back
        
        strategy._normalize_result.assert_called_once_with({"text": "test output", "cache": {"new_key": "new_value"}}, False)
        assert results == [{"text": "normalized test output", "is_partial": False}]
        assert updated_state['cache'] == {"new_key": "new_value"}


    def test_normalize_result_normal(self):
        config = get_mock_config(model_type="normal")
        strategy = NormalModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.NORMAL_MODEL)

        # Test case 1: Standard list output
        raw_output1 = [{"text": "hello world", "is_final": False}]
        normalized1 = strategy._normalize_result(raw_output1, is_final_segment_flag=False)
        assert normalized1 == [{"text": "hello world", "is_partial": True}]

        # Test case 2: Standard dict output
        raw_output2 = {"text": "final segment", "is_final": True}
        normalized2 = strategy._normalize_result(raw_output2, is_final_segment_flag=True) # is_final_segment_flag aligned with model
        assert normalized2 == [{"text": "final segment", "is_partial": False}]
        
        # Test case 3: Empty text, but final
        raw_output3 = {"text": "", "is_final": True}
        normalized3 = strategy._normalize_result(raw_output3, is_final_segment_flag=True)
        assert normalized3 == [{"text": "", "is_partial": False}]

        # Test case 4: Empty text, not final (should return empty list as per current logic)
        raw_output4 = {"text": "", "is_final": False}
        normalized4 = strategy._normalize_result(raw_output4, is_final_segment_flag=False)
        assert normalized4 == [] # Based on: `if text or actual_is_final:`

        # Test case 5: Model output indicates final, even if segment flag is false
        raw_output5 = [{"text": "model says final", "is_final": True}]
        normalized5 = strategy._normalize_result(raw_output5, is_final_segment_flag=False)
        assert normalized5 == [{"text": "model says final", "is_partial": False}]


    def test_get_model_stride_samples_normal(self):
        stride_ms = 240
        config = get_mock_config(model_type="normal", stride_ms=stride_ms)
        strategy = NormalModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.NORMAL_MODEL)
        samples = strategy.get_model_stride_samples()
        assert samples == stride_ms * 16


@pytest.mark.asyncio
class TestONNXModelStrategy:
    @patch('digitalHuman.engine.asr.streamingASR.ONNXModelStrategy.Paraformer')
    @patch('digitalHuman.engine.asr.streamingASR.ONNXModelStrategy.snapshot_download')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_load_model_direct_path(self, mock_isdir, mock_exists, mock_snapshot_download, MockParaformer):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_paraformer_instance = MagicMock()
        MockParaformer.return_value = mock_paraformer_instance

        config = get_mock_config(model_type="onnx")
        strategy = ONNXModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.ONNX_MODEL)
        model = strategy.load_model()

        mock_snapshot_download.assert_not_called()
        MockParaformer.assert_called_once_with(
            model_dir=config.ENGINE.MODEL_PATH_ONNX,
            **config.ENGINE.CHUNK_CONFIG.ONNX_MODEL.CONSTRUCTOR_ARGS.to_dict()
        )
        assert model == mock_paraformer_instance

    @patch('digitalHuman.engine.asr.streamingASR.ONNXModelStrategy.Paraformer')
    @patch('digitalHuman.engine.asr.streamingASR.ONNXModelStrategy.snapshot_download')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_load_model_snapshot_download(self, mock_isdir, mock_exists, mock_snapshot_download, MockParaformer):
        mock_exists.return_value = False # Simulate model not found locally
        mock_isdir.return_value = False
        downloaded_path = "mock/downloaded_path"
        mock_snapshot_download.return_value = downloaded_path
        mock_paraformer_instance = MagicMock()
        MockParaformer.return_value = mock_paraformer_instance

        config = get_mock_config(model_type="onnx")
        strategy = ONNXModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.ONNX_MODEL)
        model = strategy.load_model()

        mock_snapshot_download.assert_called_once_with(
            config.ENGINE.MODEL_PATH_ONNX,
            revision=config.ENGINE.MODEL_REVISION_ONNX
        )
        MockParaformer.assert_called_once_with(
            model_dir=downloaded_path,
            **config.ENGINE.CHUNK_CONFIG.ONNX_MODEL.CONSTRUCTOR_ARGS.to_dict()
        )
        assert model == mock_paraformer_instance

    def test_get_initial_stream_state_onnx(self):
        config = get_mock_config(model_type="onnx")
        strategy = ONNXModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.ONNX_MODEL)
        initial_state = strategy.get_initial_stream_state()
        assert initial_state == {'param_dict': {'cache': {}}}


    async def test_process_segment_onnx(self):
        config = get_mock_config(model_type="onnx")
        strategy = ONNXModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.ONNX_MODEL)
        
        mock_model = AsyncMock() # Mock the loaded Paraformer model instance
        mock_model.return_value = [{"text": "onnx test output", "is_final": False}] # Mock Paraformer output
        
        audio_segment_np = np.random.randn(DEFAULT_SAMPLES_PER_CHUNK_ONNX).astype(np.int16)
        initial_param_dict = {'cache': {'key': 'value'}}
        initial_stream_state = {'param_dict': initial_param_dict}
        
        strategy._normalize_result = MagicMock(return_value=[{"text": "normalized onnx output", "is_partial": True}])

        results, updated_state = await strategy.process_segment(mock_model, audio_segment_np, initial_stream_state, is_final_engine_flag=False)

        mock_model.assert_awaited_once()
        # Check args of the mocked model call
        called_args, called_kwargs = mock_model.call_args
        assert np.array_equal(called_kwargs['audio_in'], audio_segment_np)
        assert called_kwargs['param_dict']['is_final'] == False
        assert called_kwargs['param_dict']['cache'] == initial_param_dict['cache'] # param_dict is passed and mutated
        
        strategy._normalize_result.assert_called_once_with([{"text": "onnx test output", "is_final": False}], False)
        assert results == [{"text": "normalized onnx output", "is_partial": True}]
        # param_dict is mutated in-place, so updated_state should reflect that if mock_model mutated it.
        # Here, we assume mock_model does not change the structure of param_dict beyond what FunASR does.
        assert updated_state['param_dict'] is initial_param_dict # Ensure it's the same object, mutated.

    def test_normalize_result_onnx(self):
        config = get_mock_config(model_type="onnx")
        strategy = ONNXModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.ONNX_MODEL)

        # Test case 1: Standard list output
        raw_output1 = [{"text": "hello onnx", "is_final": False}]
        normalized1 = strategy._normalize_result(raw_output1, is_final_segment_flag=False)
        assert normalized1 == [{"text": "hello onnx", "is_partial": True}]

        # Test case 2: Multiple segments in list (though unusual for single chunk processing)
        raw_output2 = [
            {"text": "segment1", "is_final": False},
            {"text": "segment2", "is_final": True} # Model marks one item as final
        ]
        normalized2 = strategy._normalize_result(raw_output2, is_final_segment_flag=False) # Overall segment is not final
        assert normalized2 == [
            {"text": "segment1", "is_partial": True},
            {"text": "segment2", "is_partial": False} # respects model's "is_final"
        ]
        
        # Test case 3: Empty text, final
        raw_output3 = [{"text": "", "is_final": True}]
        normalized3 = strategy._normalize_result(raw_output3, is_final_segment_flag=True)
        assert normalized3 == [{"text": "", "is_partial": False}]


    def test_get_model_stride_samples_onnx(self):
        stride_ms = 240
        config = get_mock_config(model_type="onnx", stride_ms=stride_ms)
        strategy = ONNXModelStrategy(config.ENGINE, config.ENGINE.CHUNK_CONFIG.ONNX_MODEL)
        samples = strategy.get_model_stride_samples()
        assert samples == stride_ms * 16


@pytest.mark.asyncio
class TestStreamingASR:
    @patch('digitalHuman.engine.asr.streamingASR.NormalModelStrategy')
    def test_init_normal_model(self, MockNormalStrategy):
        mock_strategy_instance = MagicMock(spec=NormalModelStrategy) # Use spec for better mocking
        mock_strategy_instance.load_model = MagicMock(return_value=MagicMock()) # Mocked model object
        mock_strategy_instance.get_model_stride_samples = MagicMock(return_value=DEFAULT_SAMPLES_PER_CHUNK_NORMAL)
        MockNormalStrategy.return_value = mock_strategy_instance
        
        config = get_mock_config(model_type="normal")
        asr = StreamingASR(config)

        MockNormalStrategy.assert_called_once_with(config.ENGINE, config.ENGINE.CHUNK_CONFIG.NORMAL_MODEL)
        mock_strategy_instance.load_model.assert_called_once()
        mock_strategy_instance.get_model_stride_samples.assert_called_once()
        assert asr.model_handler == mock_strategy_instance
        assert asr.model is not None
        assert asr.model_stride_samples == DEFAULT_SAMPLES_PER_CHUNK_NORMAL

    def test_init_invalid_model_type(self):
        config = get_mock_config(model_type="invalid_type")
        with pytest.raises(ValueError, match="Unsupported model type: invalid_type"):
            StreamingASR(config)

    @patch('digitalHuman.engine.asr.streamingASR.ONNXModelStrategy')
    def test_init_onnx_model(self, MockONNXStrategy):
        mock_strategy_instance = MagicMock(spec=ONNXModelStrategy)
        mock_strategy_instance.load_model = MagicMock(return_value=MagicMock())
        mock_strategy_instance.get_model_stride_samples = MagicMock(return_value=DEFAULT_SAMPLES_PER_CHUNK_ONNX)
        MockONNXStrategy.return_value = mock_strategy_instance

        config = get_mock_config(model_type="onnx")
        asr = StreamingASR(config)

        MockONNXStrategy.assert_called_once_with(config.ENGINE, config.ENGINE.CHUNK_CONFIG.ONNX_MODEL)
        mock_strategy_instance.load_model.assert_called_once()
        mock_strategy_instance.get_model_stride_samples.assert_called_once()
        assert asr.model_handler == mock_strategy_instance

    async def test_init_stream(self):
        config = get_mock_config()
        # Patch the strategy directly for StreamingASR constructor
        with patch('digitalHuman.engine.asr.streamingASR.NormalModelStrategy') as MockStrategy:
            mock_strategy_instance = MockStrategy.return_value
            mock_strategy_instance.load_model.return_value = MagicMock()
            mock_strategy_instance.get_model_stride_samples.return_value = DEFAULT_SAMPLES_PER_CHUNK_NORMAL
            mock_strategy_instance.get_initial_stream_state.return_value = {'cache': 'initial'}
            
            asr = StreamingASR(config)
            stream_id = "test_stream_123"
            await asr.init_stream(stream_id)

            mock_strategy_instance.get_initial_stream_state.assert_called_once()
            assert stream_id in asr._active_streams
            assert asr._active_streams[stream_id] == {'cache': 'initial'}
            assert stream_id in asr.internal_buffer
            assert asr.internal_buffer[stream_id] == bytearray()

    async def test_process_chunk_buffering_and_delegation(self):
        config = get_mock_config(stride_ms=20) # 20ms stride = 320 samples
        stride_samples = 20 * 16 

        with patch('digitalHuman.engine.asr.streamingASR.NormalModelStrategy') as MockStrategy:
            mock_strategy_instance = MockStrategy.return_value
            mock_strategy_instance.load_model.return_value = MagicMock()
            mock_strategy_instance.get_model_stride_samples.return_value = stride_samples
            # Mock process_segment as an AsyncMock
            mock_strategy_instance.process_segment = AsyncMock(return_value=([], {'cache': 'updated'}))
            
            asr = StreamingASR(config)
            stream_id = "test_buffering_stream"
            await asr.init_stream(stream_id)

            # Send chunk smaller than stride
            chunk1_data = np.random.randint(-100, 100, size=stride_samples // 2, dtype=np.int16).tobytes()
            results1 = await asr.process_chunk(stream_id, chunk1_data, is_final_client_chunk=False)
            assert results1 == []
            mock_strategy_instance.process_segment.assert_not_called() # Not enough data yet
            assert len(asr.internal_buffer[stream_id]) == len(chunk1_data)

            # Send another chunk to meet stride
            chunk2_data = np.random.randint(-100, 100, size=stride_samples // 2, dtype=np.int16).tobytes()
            results2 = await asr.process_chunk(stream_id, chunk2_data, is_final_client_chunk=False)
            
            mock_strategy_instance.process_segment.assert_awaited_once()
            call_args = mock_strategy_instance.process_segment.call_args
            # Expected combined audio data
            expected_audio_np = np.frombuffer(chunk1_data + chunk2_data, dtype=np.int16)
            assert np.array_equal(call_args[0][1], expected_audio_np) # audio_segment_np
            assert call_args[0][3] == False # is_final_engine_flag
            assert asr._active_streams[stream_id] == {'cache': 'updated'}
            assert len(asr.internal_buffer[stream_id]) == 0 # Buffer consumed

    async def test_process_chunk_final_flag(self):
        config = get_mock_config(stride_ms=10) # 10ms stride = 160 samples
        stride_samples = 10 * 16

        with patch('digitalHuman.engine.asr.streamingASR.NormalModelStrategy') as MockStrategy:
            mock_strategy_instance = MockStrategy.return_value
            mock_strategy_instance.load_model.return_value = MagicMock()
            mock_strategy_instance.get_model_stride_samples.return_value = stride_samples
            mock_strategy_instance.process_segment = AsyncMock(return_value=([{"text":"final", "is_partial":False}], {'cache': 'final_update'}))

            asr = StreamingASR(config)
            stream_id = "test_final_flag"
            await asr.init_stream(stream_id)

            # Send one chunk of data, mark as final client chunk
            chunk_data = np.random.randint(-100, 100, size=stride_samples, dtype=np.int16).tobytes()
            results = await asr.process_chunk(stream_id, chunk_data, is_final_client_chunk=True)
            
            mock_strategy_instance.process_segment.assert_awaited_once()
            call_args = mock_strategy_instance.process_segment.call_args
            assert call_args[0][3] == True # is_final_engine_flag should be True
            assert results == [{"text":"final", "is_partial":False}]

    async def test_end_stream_empty_buffer(self):
        config = get_mock_config()
        with patch('digitalHuman.engine.asr.streamingASR.NormalModelStrategy') as MockStrategy:
            mock_strategy_instance = MockStrategy.return_value
            mock_strategy_instance.load_model.return_value = MagicMock()
            mock_strategy_instance.get_model_stride_samples.return_value = DEFAULT_SAMPLES_PER_CHUNK_NORMAL
            mock_strategy_instance.process_segment = AsyncMock(return_value=([{"text":"empty_final", "is_partial":False}], {'cache': 'final_empty_update'}))
            
            asr = StreamingASR(config)
            stream_id = "test_end_empty"
            await asr.init_stream(stream_id) # Buffer is empty

            results = await asr.end_stream(stream_id)

            # Should call process_segment with empty audio and is_final_engine_flag=True
            mock_strategy_instance.process_segment.assert_awaited_once()
            call_args = mock_strategy_instance.process_segment.call_args
            assert call_args[0][1].size == 0 # empty audio_segment_np
            assert call_args[0][3] == True    # is_final_engine_flag
            assert results == [{"text":"empty_final", "is_partial":False}]
            assert stream_id not in asr._active_streams
            assert stream_id not in asr.internal_buffer


    async def test_end_stream_with_remaining_buffer(self):
        config = get_mock_config(stride_ms=20) # stride = 320 samples
        stride_samples = 20 * 16
        with patch('digitalHuman.engine.asr.streamingASR.NormalModelStrategy') as MockStrategy:
            mock_strategy_instance = MockStrategy.return_value
            mock_strategy_instance.load_model.return_value = MagicMock()
            mock_strategy_instance.get_model_stride_samples.return_value = stride_samples
            # Let process_segment return different results based on is_final_engine_flag
            async def mock_process_segment_logic(model, audio_np, stream_state, is_final_engine_flag):
                if is_final_engine_flag:
                    return ([{"text":"final_remaining", "is_partial":False}], {'cache': 'final_remaining_update'})
                return ([{"text":"partial_remaining", "is_partial":True}], {'cache': 'partial_remaining_update'})
            mock_strategy_instance.process_segment = AsyncMock(side_effect=mock_process_segment_logic)

            asr = StreamingASR(config)
            stream_id = "test_end_remaining"
            await asr.init_stream(stream_id)

            # Add data less than a full stride
            remaining_data_bytes = np.random.randint(-100, 100, size=stride_samples // 2, dtype=np.int16).tobytes()
            await asr.process_chunk(stream_id, remaining_data_bytes, is_final_client_chunk=False)
            
            # process_chunk should not have called process_segment yet as data < stride
            mock_strategy_instance.process_segment.assert_not_called()
            
            results = await asr.end_stream(stream_id)

            # end_stream internally calls process_chunk with is_final_client_chunk=True
            # This should trigger processing of the remaining buffer
            mock_strategy_instance.process_segment.assert_awaited_once()
            call_args = mock_strategy_instance.process_segment.call_args
            expected_audio_np = np.frombuffer(remaining_data_bytes, dtype=np.int16)
            assert np.array_equal(call_args[0][1], expected_audio_np)
            assert call_args[0][3] == True # is_final_engine_flag from the call within end_stream's process_chunk
            
            assert results == [{"text":"final_remaining", "is_partial":False}]
            assert stream_id not in asr._active_streams
            assert stream_id not in asr.internal_buffer
```
