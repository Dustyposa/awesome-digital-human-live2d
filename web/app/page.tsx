"use client"

import { useEffect } from "react";
import Live2d from "./ui/home/live2d";
import Chatbot from "./ui/home/chatbot";
import StreamingASRComponent from "./ui/StreamingASR/StreamingASRComponent"; // Added
import { InteractionMode, useInteractionModeStore, useAgentModeStore, useAgentEngineSettingsStore } from "./lib/store";

export default function Home() {
  const interactionMode = useInteractionModeStore((state) => state.mode)
  const { fetchDefaultAgent } = useAgentModeStore();
  const { fetchAgentSettings } = useAgentEngineSettingsStore();
  const showCharacter = interactionMode != InteractionMode.CHATBOT;
  const showChatbot = interactionMode != InteractionMode.IMMERSIVE;

  useEffect(() => {
    fetchDefaultAgent();
    fetchAgentSettings();
  }, [fetchDefaultAgent, fetchAgentSettings])

  return (
      <div className="flex-1 overflow-auto">
        { showCharacter ? <Live2d/> : <></>}
        { showChatbot ? <Chatbot showChatHistory={true}/> : <></>}
        {/* Added StreamingASRComponent */}
        <div style={{ position: 'fixed', bottom: '20px', left: '20px', zIndex: 1000, background: 'white', padding: '10px', border: '1px solid #ccc', borderRadius: '8px' }}>
          <StreamingASRComponent />
        </div>
      </div>
  );
}
