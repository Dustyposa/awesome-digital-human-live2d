import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch, call
from fastapi.websockets import WebSocketState

# Adjust the import path according to your project structure
from digitalHuman.server.asrStreamApi import WebSocketConnectionHandler, StreamState 
from digitalHuman.engine.asr.streamingASR import StreamingASR # For type hinting/spec

@pytest.fixture
def mock_websocket():
    ws = MagicMock(spec=WebSocket) # Using spec for WebSocket
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.client_state = WebSocketState.CONNECTED # Default to connected
    return ws

@pytest.fixture
def mock_asr_engine():
    engine = MagicMock(spec=StreamingASR)
    engine.init_stream = AsyncMock()
    engine.process_chunk = AsyncMock(return_value=[]) # Default: returns empty list of results
    engine.end_stream = AsyncMock(return_value=[])   # Default: returns empty list of final results
    
    # Simulate _active_streams as a dictionary for cleanup checks
    # The actual check in _cleanup_resources is `self.stream_id in self.asr_engine._active_streams`
    # So, we need to make sure this attribute exists on the mock.
    engine._active_streams = {} 
    
    # Helper to simulate adding/removing stream_id from active_streams
    def sim_init_stream(stream_id):
        engine._active_streams[stream_id] = {} # Add to active
    def sim_end_stream(stream_id):
        if stream_id in engine._active_streams:
            del engine._active_streams[stream_id] # Remove from active
        return [] # Return empty list as per default mock

    engine.init_stream.side_effect = sim_init_stream
    engine.end_stream.side_effect = sim_end_stream
    return engine


@pytest.mark.asyncio
async def test_handler_initialization(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    assert handler.websocket == mock_websocket
    assert handler.asr_engine == mock_asr_engine
    assert handler.machine.current_state.name == StreamState.INITIALIZING.value
    assert handler.stream_id is not None

@pytest.mark.asyncio
async def test_handle_connect_transitions_to_waiting_for_start(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()
    assert handler.machine.current_state.name == StreamState.WAITING_FOR_START.value
    # _on_enter_waiting_for_start logs, doesn't send msg by default in current code

@pytest.mark.asyncio
async def test_start_stream_received_success(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect() # Initial state: WAITING_FOR_START

    start_msg_dict = {"type": "start_stream", "config": {"sample_rate": 16000}}
    await handler.handle_message(json.dumps(start_msg_dict))

    assert handler.machine.current_state.name == StreamState.STREAMING_AUDIO.value
    mock_asr_engine.init_stream.assert_called_once_with(handler.stream_id)
    mock_websocket.send_json.assert_called_once_with(
        {"type": "stream_started", "stream_id": handler.stream_id, "message": "ASR stream started."}
    )
    assert handler.client_config == {"sample_rate": 16000}

@pytest.mark.asyncio
async def test_audio_chunk_received_success(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()
    start_msg_dict = {"type": "start_stream"}
    await handler.handle_message(json.dumps(start_msg_dict)) # To STREAMING_AUDIO

    # Mock ASR engine process_chunk to return some results
    mock_asr_engine.process_chunk = AsyncMock(return_value=[
        {"text": "hello", "is_partial": True},
        {"text": "world", "is_partial": False}
    ])

    audio_data = b"some_audio_data"
    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
    chunk_msg_dict = {"type": "audio_chunk", "data": audio_b64}
    await handler.handle_message(json.dumps(chunk_msg_dict))

    assert handler.machine.current_state.name == StreamState.STREAMING_AUDIO.value # Stays in this state
    mock_asr_engine.process_chunk.assert_called_once()
    # Check that the first argument to process_chunk is handler.stream_id
    # and the second is audio_data (decoded bytes)
    call_args = mock_asr_engine.process_chunk.call_args
    assert call_args[0][0] == handler.stream_id
    assert call_args[0][1] == audio_data
    assert call_args[0][2] is False # is_final_client_chunk

    expected_calls = [
        call({"type": "intermediate_result", "stream_id": handler.stream_id, "text": "hello"}),
        call({"type": "final_result", "stream_id": handler.stream_id, "text": "world"})
    ]
    mock_websocket.send_json.assert_has_calls(expected_calls, any_order=False)

@pytest.mark.asyncio
async def test_end_stream_received_success(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()
    start_msg_dict = {"type": "start_stream"}
    await handler.handle_message(json.dumps(start_msg_dict)) # To STREAMING_AUDIO

    # Mock ASR engine end_stream
    mock_asr_engine.end_stream = AsyncMock(return_value=[{"text": "final sentence from end", "is_partial": False}])
    
    end_msg_dict = {"type": "end_stream"}
    await handler.handle_message(json.dumps(end_msg_dict))

    # State flow: STREAMING_AUDIO -> PROCESSING_FINAL_CHUNK (after _process_final_chunks) -> CLOSING (after _send_stream_ended)
    assert handler.machine.current_state.name == StreamState.CLOSING.value
    mock_asr_engine.end_stream.assert_called_once_with(handler.stream_id)
    
    # Check messages sent: final_result from end_stream, then stream_ended
    expected_calls = [
        call({"type": "stream_started", "stream_id": handler.stream_id, "message": "ASR stream started."}), # From start
        call({"type": "final_result", "stream_id": handler.stream_id, "text": "final sentence from end"}),
        call({"type": "stream_ended", "stream_id": handler.stream_id, "message": "ASR stream processing complete."})
    ]
    mock_websocket.send_json.assert_has_calls(expected_calls, any_order=False)


@pytest.mark.asyncio
async def test_audio_chunk_in_waiting_for_start_state(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect() # State: WAITING_FOR_START

    chunk_msg_dict = {"type": "audio_chunk", "data": "some_data"}
    await handler.handle_message(json.dumps(chunk_msg_dict))

    assert handler.machine.current_state.name == StreamState.ERROR.value
    mock_websocket.send_json.assert_called_once_with(
        {"type": "error", "stream_id": handler.stream_id, "error_code": 4000, "message": f"Invalid message type 'audio_chunk' or state '{StreamState.WAITING_FOR_START.value}'"}
    )
    # _send_error_and_close_ws_connection also calls _cleanup_resources and ws.close()
    mock_asr_engine.end_stream.assert_not_called() # Stream was not init
    mock_websocket.close.assert_called_once_with(code=1011)


@pytest.mark.asyncio
async def test_asr_engine_init_stream_error(mock_websocket, mock_asr_engine):
    # Reset side_effect for init_stream to raise error
    mock_asr_engine.init_stream.side_effect = Exception("Init stream failed")
    # Keep _active_streams as it was, as init_stream fails before adding to it
    original_active_streams = dict(mock_asr_engine._active_streams) 

    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()

    start_msg_dict = {"type": "start_stream"}
    await handler.handle_message(json.dumps(start_msg_dict)) # Attempt to start

    assert handler.machine.current_state.name == StreamState.ERROR.value
    mock_websocket.send_json.assert_called_once_with(
        {"type": "error", "stream_id": handler.stream_id, "error_code": 5000, "message": "Server error processing message: Init stream failed"}
    )
    assert mock_asr_engine._active_streams == original_active_streams # Ensure no partial state change
    mock_websocket.close.assert_called_once_with(code=1011)


@pytest.mark.asyncio
async def test_invalid_json_message(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()

    await handler.handle_message("this is not json")

    assert handler.machine.current_state.name == StreamState.ERROR.value
    mock_websocket.send_json.assert_called_once_with(
        {"type": "error", "stream_id": handler.stream_id, "error_code": 4001, "message": "Invalid JSON format."}
    )
    mock_websocket.close.assert_called_once_with(code=1011)

@pytest.mark.asyncio
async def test_invalid_base64_in_audio_chunk(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()
    start_msg_dict = {"type": "start_stream"}
    await handler.handle_message(json.dumps(start_msg_dict)) # To STREAMING_AUDIO

    chunk_msg_dict = {"type": "audio_chunk", "data": "this is not base64!"}
    await handler.handle_message(json.dumps(chunk_msg_dict))
    
    assert handler.machine.current_state.name == StreamState.ERROR.value
    # The error message from b64decode can vary, so we check for a substring
    sent_error_msg = mock_websocket.send_json.call_args[0][0]["message"]
    assert "Invalid base64 audio data" in sent_error_msg
    mock_websocket.close.assert_called_once_with(code=1011)


@pytest.mark.asyncio
async def test_start_stream_in_streaming_audio_state(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()
    # Send initial start_stream to transition to STREAMING_AUDIO
    initial_start_msg = {"type": "start_stream", "config": {"sample_rate": 16000}}
    await handler.handle_message(json.dumps(initial_start_msg))
    assert handler.machine.current_state.name == StreamState.STREAMING_AUDIO.value

    # Ensure init_stream was called once for the initial setup
    mock_asr_engine.init_stream.assert_called_once()
    # Reset send_json mock to only capture the error message for the second start_stream
    mock_websocket.send_json.reset_mock() 

    # Send another start_stream message
    second_start_msg = {"type": "start_stream", "config": {"sample_rate": 32000}}
    await handler.handle_message(json.dumps(second_start_msg))

    assert handler.machine.current_state.name == StreamState.ERROR.value
    # init_stream should NOT be called again
    mock_asr_engine.init_stream.assert_called_once() 
    mock_websocket.send_json.assert_called_once_with(
        {"type": "error", "stream_id": handler.stream_id, "error_code": 4000, "message": f"Invalid message type 'start_stream' or state '{StreamState.STREAMING_AUDIO.value}'"}
    )
    mock_websocket.close.assert_called_once_with(code=1011)


@pytest.mark.asyncio
async def test_asr_engine_end_stream_error(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()
    start_msg_dict = {"type": "start_stream"}
    await handler.handle_message(json.dumps(start_msg_dict)) # To STREAMING_AUDIO
    assert handler.stream_id in mock_asr_engine._active_streams # Stream is active

    # Configure end_stream to raise an error, but also ensure it still cleans up from _active_streams
    # for the purpose of this test's verification of cleanup path.
    def end_stream_error_side_effect(stream_id):
        if stream_id in mock_asr_engine._active_streams:
            del mock_asr_engine._active_streams[stream_id]
        raise Exception("End stream failed")
    mock_asr_engine.end_stream.side_effect = end_stream_error_side_effect
    
    # Reset send_json mock to only capture the error message from this specific operation
    mock_websocket.send_json.reset_mock()

    end_msg_dict = {"type": "end_stream"}
    await handler.handle_message(json.dumps(end_msg_dict)) # Attempt to end stream

    assert handler.machine.current_state.name == StreamState.ERROR.value
    # Error message from _process_final_chunks
    mock_websocket.send_json.assert_called_once_with(
        {"type": "error", "stream_id": handler.stream_id, "error_code": 5000, "message": "Error during final processing: End stream failed"}
    )
    assert not (handler.stream_id in mock_asr_engine._active_streams) # Should still be cleaned
    mock_websocket.close.assert_called_once_with(code=1011)


@pytest.mark.asyncio
async def test_client_disconnect_while_streaming(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()
    start_msg_dict = {"type": "start_stream"}
    await handler.handle_message(json.dumps(start_msg_dict)) # To STREAMING_AUDIO
    
    # Simulate stream being active in engine
    assert handler.stream_id in mock_asr_engine._active_streams

    await handler.handle_disconnect()

    assert handler.machine.current_state.name == StreamState.CLOSING.value
    # end_stream side effect in mock_asr_engine will remove from _active_streams
    # So we check that it was called, which implies it was cleaned.
    # The number of calls to end_stream depends on how many times _cleanup_resources is hit.
    # The important part is that it *is* called if the stream was active.
    assert not (handler.stream_id in mock_asr_engine._active_streams) # Check it was removed
    # The mock_asr_engine.end_stream is called by _cleanup_resources.
    # We need to check if _cleanup_resources itself was called via the close_connection transition.
    # To do this properly, we might need to spy on _cleanup_resources or ensure its effects.
    # The current check `assert not (handler.stream_id in mock_asr_engine._active_streams)` is a good indicator.
    mock_asr_engine.end_stream.assert_called_once_with(handler.stream_id)


@pytest.mark.asyncio
async def test_client_disconnect_before_start(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect() # State: WAITING_FOR_START
    
    initial_active_streams_count = len(mock_asr_engine._active_streams)
    await handler.handle_disconnect()

    assert handler.machine.current_state.name == StreamState.CLOSING.value
    mock_asr_engine.end_stream.assert_not_called() # Because init_stream was not called
    assert len(mock_asr_engine._active_streams) == initial_active_streams_count


@pytest.mark.asyncio
async def test_resource_cleanup_on_error_state(mock_websocket, mock_asr_engine):
    handler = WebSocketConnectionHandler(mock_websocket, mock_asr_engine)
    await handler.handle_connect()
    start_msg_dict = {"type": "start_stream"}
    await handler.handle_message(json.dumps(start_msg_dict)) # To STREAMING_AUDIO
    assert handler.stream_id in mock_asr_engine._active_streams # Stream is active

    # Trigger an error that leads to ERROR state
    mock_asr_engine.process_chunk.side_effect = Exception("ASR processing error")
    chunk_msg_dict = {"type": "audio_chunk", "data": base64.b64encode(b"audio").decode('utf-8')}
    await handler.handle_message(json.dumps(chunk_msg_dict))

    assert handler.machine.current_state.name == StreamState.ERROR.value
    # _cleanup_resources is called by _send_error_and_close_ws_connection
    # end_stream mock will remove from _active_streams
    assert not (handler.stream_id in mock_asr_engine._active_streams)
    mock_asr_engine.end_stream.assert_called_once_with(handler.stream_id)
    mock_websocket.close.assert_called_once_with(code=1011)
```
