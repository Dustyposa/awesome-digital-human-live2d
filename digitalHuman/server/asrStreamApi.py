import asyncio
import json
import logging
import uuid
import base64
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState # For checking client_state

from transitions.extensions.asyncio import AsyncMachine

# Project-specific imports
from digitalHuman.engine.asr.streamingASR import StreamingASR
from digitalHuman.engine.enginePool import EnginePool, EngineType
# from digitalHuman.utils import config as global_config # Not directly used in this file if EnginePool is pre-configured

# Logger Setup
logger = logging.getLogger(__name__)

# APIRouter
router = APIRouter()

# Accessing the Engine
# Assuming EnginePool is initialized at app startup and has the StreamingASR engine configured.
engine_pool = EnginePool()
# Note: EnginePool.setup(global_config) should have been called elsewhere, e.g., in main.py or app startup.

class StreamState(Enum):
    INITIALIZING = 'initializing'
    WAITING_FOR_START = 'waiting_for_start'
    STREAMING_AUDIO = 'streaming_audio'
    PROCESSING_FINAL_CHUNK = 'processing_final_chunk'
    CLOSING = 'closing'
    ERROR = 'error'

class WebSocketConnectionHandler:
    def __init__(self, websocket: WebSocket, asr_engine: StreamingASR):
        self.websocket = websocket
        self.asr_engine = asr_engine
        self.stream_id = str(uuid.uuid4())
        self.client_config = {}
        self.error_message = "Unknown error" # Default error message

        states = [s.value for s in StreamState]
        
        self.machine = AsyncMachine(
            model=self,
            states=states,
            initial=StreamState.INITIALIZING.value,
            send_event=True,
            auto_transitions=False,
            queued=True 
        )

        # Transitions
        self.machine.add_transition(trigger='connection_accepted', source=StreamState.INITIALIZING.value, dest=StreamState.WAITING_FOR_START.value, after='_on_enter_waiting_for_start')
        self.machine.add_transition(trigger='start_stream_received', source=StreamState.WAITING_FOR_START.value, dest=StreamState.STREAMING_AUDIO.value, before='_prepare_stream', after='_send_stream_started')
        self.machine.add_transition(trigger='audio_chunk_received', source=StreamState.STREAMING_AUDIO.value, dest='=', after='_process_audio_chunk')
        self.machine.add_transition(trigger='end_stream_received', source=StreamState.STREAMING_AUDIO.value, dest=StreamState.PROCESSING_FINAL_CHUNK.value, after='_process_final_chunks')
        self.machine.add_transition(trigger='final_processing_complete', source=StreamState.PROCESSING_FINAL_CHUNK.value, dest=StreamState.CLOSING.value, after='_send_stream_ended')
        self.machine.add_transition(trigger='close_connection', source='*', dest=StreamState.CLOSING.value, before='_cleanup_resources') # before to ensure cleanup happens first
        self.machine.add_transition(trigger='error_occurred', source='*', dest=StreamState.ERROR.value, before='_set_error_details', after='_send_error_and_close_ws_connection')


    async def _on_enter_waiting_for_start(self, event_data):
        logger.info(f"[{self.stream_id}] WebSocket connection accepted, state: {self.state}. Waiting for start_stream.")
        # Optional: await self.websocket.send_json({"type": "info", "message": "Connection accepted, please send start_stream"})

    async def _prepare_stream(self, event_data):
        self.client_config = event_data.kwargs.get("config", {})
        logger.info(f"[{self.stream_id}] Received start_stream with config: {self.client_config}. Initializing ASR stream.")
        await self.asr_engine.init_stream(self.stream_id)

    async def _send_stream_started(self, event_data):
        logger.info(f"[{self.stream_id}] ASR stream initialized. Sending stream_started.")
        await self.websocket.send_json({"type": "stream_started", "stream_id": self.stream_id, "message": "ASR stream started."})

    async def _process_audio_chunk(self, event_data):
        audio_b64 = event_data.kwargs.get("data")
        if not audio_b64:
            logger.warning(f"[{self.stream_id}] Received empty audio_chunk data.")
            # Optionally send a specific error or ignore
            return
        try:
            audio_bytes = base64.b64decode(audio_b64)
            logger.debug(f"[{self.stream_id}] Processing audio chunk of size: {len(audio_bytes)}")
            results = await self.asr_engine.process_chunk(self.stream_id, audio_bytes, is_final_client_chunk=False)
            for res in results:
                msg_type = "intermediate_result" if res.get("is_partial") else "final_result"
                await self.websocket.send_json({"type": msg_type, "stream_id": self.stream_id, "text": res.get("text")})
        except base64.binascii.Error as b64e:
            logger.error(f"[{self.stream_id}] Base64 decoding error: {b64e}", exc_info=True)
            await self.error_occurred(error_message=f"Invalid base64 audio data: {str(b64e)}")
        except Exception as e:
            logger.error(f"[{self.stream_id}] Error processing audio chunk: {e}", exc_info=True)
            await self.error_occurred(error_message=f"Error processing audio chunk: {str(e)}")

    async def _process_final_chunks(self, event_data):
        logger.info(f"[{self.stream_id}] Processing final audio chunks after end_stream received.")
        try:
            final_results = await self.asr_engine.end_stream(self.stream_id)
            for res in final_results:
                await self.websocket.send_json({"type": "final_result", "stream_id": self.stream_id, "text": res.get("text")})
            await self.final_processing_complete()
        except Exception as e:
            logger.error(f"[{self.stream_id}] Error during final chunk processing: {e}", exc_info=True)
            await self.error_occurred(error_message=f"Error during final processing: {str(e)}")

    async def _send_stream_ended(self, event_data):
        logger.info(f"[{self.stream_id}] Sending stream_ended.")
        await self.websocket.send_json({"type": "stream_ended", "stream_id": self.stream_id, "message": "ASR stream processing complete."})
        # The 'close_connection' transition is now separate and will be called by FastAPI endpoint's finally or disconnect
        # Or it can be triggered here if we want the state machine to drive the final websocket close
        # For now, let it be handled by endpoint exit or explicit disconnect.

    async def _set_error_details(self, event_data):
        self.error_message = event_data.kwargs.get("error_message", "Unknown error")
        logger.error(f"[{self.stream_id}] Error state '{self.state}' entered: {self.error_message}")

    async def _send_error_and_close_ws_connection(self, event_data):
        # Ensure error message is sent before closing
        if self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.websocket.send_json({"type": "error", "stream_id": self.stream_id, "error_code": event_data.kwargs.get("error_code", 5000) ,"message": self.error_message})
            except Exception as e:
                logger.error(f"[{self.stream_id}] Failed to send error message to client: {e}", exc_info=True)
        
        # Now trigger actual cleanup and resource release
        await self._cleanup_resources(event_data, from_error_path=True) # Pass a flag

        if self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                logger.info(f"[{self.stream_id}] Closing WebSocket connection due to error.")
                await self.websocket.close(code=1011) # Internal error, or a more specific code
            except Exception as e:
                logger.error(f"[{self.stream_id}] Error trying to close WebSocket connection: {e}", exc_info=True)


    async def _cleanup_resources(self, event_data, from_error_path=False):
        logger.info(f"[{self.stream_id}] Cleaning up resources. Current state: {self.state}. From error path: {from_error_path}")
        if self.stream_id in self.asr_engine._active_streams:
            try:
                logger.info(f"[{self.stream_id}] Calling end_stream on ASR engine for cleanup.")
                await self.asr_engine.end_stream(self.stream_id) # This also removes from _active_streams
            except Exception as e:
                logger.error(f"[{self.stream_id}] Error during ASR engine resource cleanup: {e}", exc_info=True)
        else:
            logger.info(f"[{self.stream_id}] ASR stream already ended or not initialized, no engine cleanup needed by this call.")
        
        logger.info(f"[{self.stream_id}] Resource cleanup finished.")
        # WebSocket closing is generally handled by the FastAPI endpoint or by _send_error_and_close_ws_connection

    async def handle_connect(self):
        await self.connection_accepted()

    async def handle_disconnect(self):
        logger.info(f"[{self.stream_id}] Client disconnected. Current state: {self.state}")
        if self.state not in [StreamState.CLOSING.value, StreamState.ERROR.value]:
            logger.info(f"[{self.stream_id}] Transitioning to close_connection due to disconnect.")
            await self.close_connection() # This will trigger _cleanup_resources
        else:
            logger.info(f"[{self.stream_id}] Already in a terminal state or error state. Cleanup should have run or will run.")


    async def handle_message(self, raw_message: str):
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type")
            logger.debug(f"[{self.stream_id}] Received message type: {msg_type} in state: {self.state}")

            if msg_type == "start_stream": # No need for machine.can_event(), just trigger
                await self.start_stream_received(config=message.get("config", {}))
            elif msg_type == "audio_chunk":
                await self.audio_chunk_received(data=message.get("data"))
            elif msg_type == "end_stream":
                await self.end_stream_received()
            else:
                logger.warning(f"[{self.stream_id}] Unknown message type: {msg_type} or invalid in state {self.state}")
                await self.error_occurred(error_message=f"Invalid message type '{msg_type}' or state '{self.state}'", error_code=4000)
        except json.JSONDecodeError:
            logger.error(f"[{self.stream_id}] Invalid JSON message received.", exc_info=True)
            await self.error_occurred(error_message="Invalid JSON format.", error_code=4001)
        except Exception as e: # Catch other errors during message processing
            logger.error(f"[{self.stream_id}] Error handling message: {e}", exc_info=True)
            await self.error_occurred(error_message=f"Server error processing message: {str(e)}", error_code=5000)


# The WebSocket path in the decorator should match the one specified in the task,
# considering any prefixing done in api.py. If api.py uses prefix="/adh/ws/asr",
# and the desired final path is "/adh/ws/asr/streaming", then this should be "/streaming".
# The initial subtask for asrStreamApi.py used "/ws/v0/asr/streaming".
# Let's assume the combined prefixing has been handled and this is the final part of the path.
@router.websocket("/streaming") # Or "/ws/v0/asr/streaming" if no prefixing in api.py for this part
async def websocket_streaming_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    asr_engine_instance: StreamingASR = None
    try:
        # Using synchronous getEngine as established
        asr_engine_instance = engine_pool.getEngine(EngineType.ASR, "StreamingASR")
        if not asr_engine_instance or not isinstance(asr_engine_instance, StreamingASR):
            logger.error("StreamingASR engine not available or incorrect type.")
            await websocket.send_json({"type": "error", "stream_id": "N/A", "error_code": 5001, "message": "ASR engine not available."})
            await websocket.close(code=1011)
            return
    except Exception as e: # Catch errors during engine acquisition
        logger.error(f"Failed to get ASR engine: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "stream_id": "N/A", "error_code": 5002, "message": "Failed to initialize ASR service."})
        await websocket.close(code=1011)
        return

    handler = WebSocketConnectionHandler(websocket, asr_engine_instance)
    stream_id_for_log = handler.stream_id # Get stream_id for logging before any potential error in handler
    logger.info(f"[{stream_id_for_log}] WebSocket handler created. Initializing connection with state machine.")
    await handler.handle_connect()

    try:
        # Loop as long as the WebSocket is connected and the state machine is not in a terminal state
        while handler.state not in [StreamState.CLOSING.value, StreamState.ERROR.value] and \
              websocket.client_state == WebSocketState.CONNECTED:
            raw_message = await websocket.receive_text()
            await handler.handle_message(raw_message)
            
    except WebSocketDisconnect:
        logger.info(f"[{handler.stream_id}] WebSocket disconnected by client. Current state: {handler.state}")
        await handler.handle_disconnect()
    except Exception as e:
        logger.error(f"[{handler.stream_id}] Unhandled exception in WebSocket endpoint: {e}", exc_info=True)
        if handler.state not in [StreamState.CLOSING.value, StreamState.ERROR.value]:
            try:
                await handler.error_occurred(error_message=f"Unhandled server error: {str(e)}", error_code=5003)
            except Exception as ie: # If error_occurred itself fails
                 logger.critical(f"[{handler.stream_id}] Failed to transition to error state: {ie}", exc_info=True)
    finally:
        logger.info(f"[{handler.stream_id}] Exiting WebSocket endpoint. Final state: {handler.state}.")
        # Ensure cleanup if the machine didn't naturally reach CLOSING or ERROR,
        # or if an unhandled exception occurred before proper cleanup.
        if handler.state not in [StreamState.CLOSING.value, StreamState.ERROR.value]:
            logger.warning(f"[{handler.stream_id}] Forcing cleanup for stream due to unexpected exit. State: {handler.state}")
            # Manually trigger cleanup if not already in a terminal state that runs it.
            # _cleanup_resources is idempotent regarding asr_engine.end_stream if stream_id not in _active_streams.
            await handler._cleanup_resources(event_data=None) # event_data might not be available here
        
        # Ensure WebSocket is closed if not already
        if websocket.client_state == WebSocketState.CONNECTED:
            logger.info(f"[{handler.stream_id}] Ensuring WebSocket is closed in finally block.")
            await websocket.close(code=1001) # Going away
        
        logger.info(f"[{handler.stream_id}] WebSocket endpoint processing finished.")
```
