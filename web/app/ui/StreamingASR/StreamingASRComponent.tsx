"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { float32ToInt16, int16ArrayToBase64, resampleBuffer } from '@/app/lib/audioUtils';
// import { useAppStore } from '@/app/lib/store'; // Example if using Zustand for global state

// Define types for messages
interface ServerMessage {
  type: string;
  stream_id?: string;
  text?: string;
  message?: string;
  error_code?: number;
}

const StreamingASRComponent: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Idle");
  const [intermediateTranscript, setIntermediateTranscript] = useState("");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioBufferRef = useRef<Float32Array[]>([]); // To buffer audio chunks

  const TARGET_SAMPLE_RATE = 16000;
  const CHUNK_DURATION_MS = 240;
  const SAMPLES_PER_CHUNK = (TARGET_SAMPLE_RATE * CHUNK_DURATION_MS) / 1000; // 16000 * 0.240 = 3840 samples

  // Placeholder for WebSocket URL - replace with actual server URL
  const WEBSOCKET_URL = `ws://${typeof window !== 'undefined' ? window.location.host : 'localhost:8000'}/adh/ws/asr/streaming`;

  const connectWebSocket = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      console.log("WebSocket already open.");
      return;
    }

    setStatusMessage("Connecting to WebSocket...");
    socketRef.current = new WebSocket(WEBSOCKET_URL);

    socketRef.current.onopen = () => {
      setStatusMessage("WebSocket connected. Sending start_stream...");
      const startStreamMsg = {
        type: "start_stream",
        config: {
          language: "zh-cn", // Or make configurable
          sample_rate: TARGET_SAMPLE_RATE,
          encoding: "pcm_s16le",
          chunk_size_ms: CHUNK_DURATION_MS
        }
      };
      socketRef.current?.send(JSON.stringify(startStreamMsg));
      // Start audio capture will be triggered after 'stream_started' or here if no explicit server ack needed before capture
    };

    socketRef.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data as string) as ServerMessage;
        console.log("Received message from server:", message);

        switch (message.type) {
          case "stream_started":
            setStatusMessage("Listening...");
            setIsRecording(true); // Now safe to set isRecording true
            startAudioCapture(); // Start capturing audio after server confirms
            break;
          case "intermediate_result":
            setIntermediateTranscript(message.text || "");
            break;
          case "final_result":
            setFinalTranscript((prev) => prev + (message.text || "") + " ");
            setIntermediateTranscript(""); // Clear intermediate
            break;
          case "stream_ended":
            setStatusMessage("Stream ended by server.");
            stopStreaming(false); // false because server initiated the end
            break;
          case "error":
            setError(`Server error: ${message.message} (Code: ${message.error_code})`);
            setStatusMessage(`Error: ${message.message}`);
            stopStreaming(false); // Stop on error
            break;
          default:
            console.warn("Unknown message type from server:", message.type);
        }
      } catch (e) {
        console.error("Failed to parse server message:", event.data, e);
        setError("Failed to parse server message.");
      }
    };

    socketRef.current.onerror = (event) => {
      console.error("WebSocket error:", event);
      setError("WebSocket error occurred. Check console for details.");
      setStatusMessage("WebSocket error.");
      setIsRecording(false); // Ensure recording stops
    };

    socketRef.current.onclose = (event) => {
      console.log("WebSocket closed:", event.reason, event.code);
      setStatusMessage(`WebSocket closed: ${event.reason || "Unknown reason"}`);
      // setIsRecording(false); // Already handled by stopStreaming or error
      if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
      }
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        // audioContextRef.current.close(); // Close audio context if necessary
      }
      // socketRef.current = null; // Let it be handled by start/stop logic
    };
  }, [WEBSOCKET_URL]); 

  const startAudioCapture = async () => {
    if (audioContextRef.current && audioContextRef.current.state === 'running') {
        console.log("Audio capture already running.");
        return;
    }
    setStatusMessage("Initializing audio...");
    audioBufferRef.current = []; // Clear buffer

    try {
      // Ensure existing AudioContext is closed or reused properly
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
          sampleRate: TARGET_SAMPLE_RATE // Request target sample rate
        });
      } else if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      
      // Verify actual sample rate
      const currentSampleRate = audioContextRef.current.sampleRate;
      if (currentSampleRate !== TARGET_SAMPLE_RATE) {
          console.warn(`AudioContext sample rate is ${currentSampleRate}Hz, not ${TARGET_SAMPLE_RATE}Hz. Resampling will be applied.`);
          setStatusMessage(`Audio Resampling: ${currentSampleRate}Hz -> ${TARGET_SAMPLE_RATE}Hz`);
      }

      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: TARGET_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
        },
      });

      const source = audioContextRef.current.createMediaStreamSource(mediaStreamRef.current);
      // Buffer size for ScriptProcessorNode.
      // For 240ms chunks at 16kHz: 0.240 * 16000 = 3840 samples.
      // ScriptProcessorNode buffer sizes must be powers of 2, e.g., 256, 512, 1024, 2048, 4096, 8192, 16384.
      // We use 4096, which is close to 3840. We'll buffer internally to get exact 3840 sample chunks.
      const bufferSize = 4096; 
      scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(bufferSize, 1, 1);

      scriptProcessorRef.current.onaudioprocess = (event: AudioProcessingEvent) => {
        if (!isRecording || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        const inputData = event.inputBuffer.getChannelData(0); // Get Float32Array
        let processedData = inputData.slice(); // Buffer a copy

        // Resample if necessary
        if (audioContextRef.current && audioContextRef.current.sampleRate !== TARGET_SAMPLE_RATE) {
          processedData = resampleBuffer(processedData, audioContextRef.current.sampleRate, TARGET_SAMPLE_RATE);
        }
        
        audioBufferRef.current.push(processedData); 

        // Calculate accumulated samples based on TARGET_SAMPLE_RATE for chunking logic
        let accumulatedTargetSamples = audioBufferRef.current.reduce((sum, arr) => sum + arr.length, 0);
        
        while (accumulatedTargetSamples >= SAMPLES_PER_CHUNK) {
          const combinedBufferForChunk = new Float32Array(SAMPLES_PER_CHUNK);
          let offset = 0;
          let samplesToProcessThisIteration = SAMPLES_PER_CHUNK;

          // Reconstruct the chunk from buffered (potentially resampled) segments
          while(samplesToProcessThisIteration > 0 && audioBufferRef.current.length > 0) {
            const currentSegment = audioBufferRef.current[0];
            const takeFromSegment = Math.min(samplesToProcessThisIteration, currentSegment.length);
            
            combinedBufferForChunk.set(currentSegment.subarray(0, takeFromSegment), offset);
            offset += takeFromSegment;
            samplesToProcessThisIteration -= takeFromSegment;

            if (takeFromSegment === currentSegment.length) {
              audioBufferRef.current.shift(); // Remove used segment
            } else {
              audioBufferRef.current[0] = currentSegment.subarray(takeFromSegment); // Keep remaining part
            }
          }
          
          const int16Chunk = float32ToInt16(combinedBufferForChunk);
          const base64Chunk = int16ArrayToBase64(int16Chunk);

          if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({ type: "audio_chunk", data: base64Chunk }));
            console.log(`Sent audio_chunk of ${int16Chunk.length} samples (target ${TARGET_SAMPLE_RATE}Hz)`);
          } else {
            console.warn("WebSocket not open, cannot send audio_chunk.");
            // Optionally stop recording or buffer chunks if desired. For now, just logs.
          }
          accumulatedTargetSamples = audioBufferRef.current.reduce((sum, arr) => sum + arr.length, 0);
        }
      };

      source.connect(scriptProcessorRef.current);
      scriptProcessorRef.current.connect(audioContextRef.current.destination); // Connect to output, though not strictly necessary for processing only

      // setIsRecording(true); // Moved to be set after server confirms 'stream_started'
      setStatusMessage("Audio capture started. Listening...");
      setError(null);

    } catch (err) {
      console.error("Error starting audio capture:", err);
      setError(`Error starting audio capture: ${(err as Error).message}`);
      setStatusMessage("Error: Could not start audio capture.");
      setIsRecording(false);
    }
  };

  const stopStreaming = useCallback((clientInitiated = true) => {
    console.log("Stopping stream. Client initiated:", clientInitiated);
    setIsRecording(false); // Set immediately to stop onaudioprocess loop logic
    setStatusMessage("Stopping stream...");

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect(); // Disconnect ScriptProcessorNode
      // scriptProcessorRef.current = null; // Not strictly needed to null it here
    }
    if (audioContextRef.current && audioContextRef.current.state === "running") {
      // Consider suspending or closing AudioContext. Suspending is less destructive if starting again.
      // audioContextRef.current.suspend();
      // For a full stop, closing might be better if component unmounts or won't restart soon.
      // await audioContextRef.current.close(); 
      // audioContextRef.current = null;
    }
    
    audioBufferRef.current = []; // Clear any remaining buffered audio

    if (clientInitiated && socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      console.log("Sending end_stream message to server.");
      socketRef.current.send(JSON.stringify({ type: "end_stream" }));
      setStatusMessage("Stream ended by client. Waiting for final server response.");
    } else if (!clientInitiated) {
        // If server initiated, we might already be in a closed/error state.
        // If socket is still open, close it.
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            socketRef.current.close();
        }
    }


    // Don't nullify socketRef.current here if onclose needs it.
    // It will be handled by onclose or when starting a new connection.
  }, []);


  const handleStartStreaming = () => {
    if (isRecording) return;
    setError(null);
    setFinalTranscript("");
    setIntermediateTranscript("");
    connectWebSocket(); 
    // startAudioCapture() will be called by WebSocket onmessage('stream_started')
  };

  const handleStopStreaming = () => {
    if (!isRecording && (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN)) {
        // If not recording and socket is not even open or already closed, just reset UI.
        setStatusMessage("Not recording and socket is closed.");
        return;
    }
    stopStreaming(true); // true because client (button click) initiated the stop
  };

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      console.log("StreamingASRComponent unmounting. Cleaning up...");
      stopStreaming(true); // Ensure resources are released
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close().catch(e => console.error("Error closing AudioContext on unmount", e));
        audioContextRef.current = null;
      }
    };
  }, [stopStreaming]); // stopStreaming is memoized with useCallback

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Streaming ASR Test</h2>
      <div>
        <button onClick={handleStartStreaming} disabled={isRecording}>
          Start Streaming
        </button>
        <button onClick={handleStopStreaming} disabled={!isRecording && (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN)}>
          Stop Streaming
        </button>
      </div>
      <div style={{ marginTop: '20px' }}>
        <strong>Status:</strong> <span id="status">{statusMessage}</span>
      </div>
      {error && (
        <div style={{ color: 'red', marginTop: '10px' }}>
          <strong>Error:</strong> {error}
        </div>
      )}
      <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px', minHeight: '100px' }}>
        <h4>Live Transcript:</h4>
        <p id="intermediate-transcript" style={{ color: 'gray' }}>{intermediateTranscript}</p>
        <p id="final-transcript">{finalTranscript}</p>
      </div>
    </div>
  );
};

export default StreamingASRComponent;

// Notes on potential improvements/issues:
// 1. Resampling: If audioContext.sampleRate is not TARGET_SAMPLE_RATE, resampling is needed.
//    This PoC currently proceeds with a warning. Robust solution requires a resampling library or manual implementation.
// 2. ScriptProcessorNode is deprecated. AudioWorklet is the modern alternative but more complex to set up (requires a separate worklet file).
//    For this PoC, ScriptProcessorNode is used for simplicity.
// 3. Error handling for WebSocket connection can be more granular (e.g., retries).
// 4. UI can be made more sophisticated (e.g., visual feedback for recording).
// 5. Configuration (language, WebSocket URL) could be externalized.
```
