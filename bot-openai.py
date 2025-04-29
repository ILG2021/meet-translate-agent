#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import asyncio
import os

import aiohttp
import yaml
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai_realtime_beta import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    OpenAIRealtimeBetaLLMService,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.transports.services.daily import DailyTransport, DailyParams

from runner import configure
from util import read_prompt_from_yaml

load_dotenv(override=True)

async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Set up Daily transport with specific audio/video parameters for Gemini
        transport = DailyTransport(
            room_url,
            token,
            "Translator",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        session_properties = SessionProperties(
            input_audio_transcription=InputAudioTranscription(),
            # Set openai TurnDetection parameters. Not setting this at all will turn it
            # on by default
            turn_detection=SemanticTurnDetection(),
            # Or set to False to disable openai turn detection and use transport VAD
            # turn_detection=False,
            input_audio_noise_reduction=InputAudioNoiseReduction(type="near_field"),
            # tools=tools,
            instructions=read_prompt_from_yaml("./config.yaml"),
        )

        llm = OpenAIRealtimeBetaLLMService(
            model="gpt-4o-mini-realtime-preview-2024-12-17",
            api_key=os.getenv("OPENAI_API_KEY"),
            session_properties=session_properties,
            start_audio_paused=False,
        )

        # you can either register a single function for all function calls, or specific functions
        # llm.register_function(None, fetch_weather_from_api)
        # Create a standard OpenAI LLM context object using the normal messages format. The
        # OpenAIRealtimeBetaLLMService will convert this internally to messages that the
        # openai WebSocket API can understand.
        context = OpenAILLMContext(
            [{"role": "user", "content": "Say hello!"}],
        )

        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                context_aggregator.user(),
                llm,  # LLM
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())