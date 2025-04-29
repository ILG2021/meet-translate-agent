#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
from dataclasses import dataclass

import aiohttp
import google.ai.generativelanguage as glm
import lingua
from dotenv import load_dotenv
from lingua import LanguageDetectorBuilder
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame, TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig
from pipecat.services.google.llm import GoogleLLMService, GoogleLLMContext
from pipecat.services.google.tts import GoogleTTSService, language_to_google_tts_language
from pipecat.transcriptions.language import Language
from pipecat.transports.services.daily import DailyTransport, DailyParams

from runner import configure
from util import read_prompt_from_yaml

load_dotenv(override=True)

detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()


@dataclass
class MagicDemoTranscriptionFrame(Frame):
    text: str


class UserAudioCollector(FrameProcessor):
    def __init__(self,context, context_aggregator_user):
        super().__init__()
        self._context = context
        self._context_aggregator_user = context_aggregator_user
        self._audio_frames = []
        self._start_secs = 0.2  # this should match VAD start_secs (hardcoding for now)
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # We could gracefully handle both audio input and text/transcription input ...
            # but let's leave that as an exercise to the reader. :-)
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            # 添加音频到上下文
            self._context.add_audio_frames_message(audio_frames=self._audio_frames)
            # 推送 LLMContextFrame 给 llm
            await self._context_aggregator_user.push_frame(
                self._context_aggregator_user.get_context_frame()
            )
            # 清空缓冲区并重置上下文（无历史）
            self._audio_frames = []
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_frames.append(frame)
            else:
                # Append the audio frame to our buffer. Treat the buffer as a ring buffer, dropping the oldest
                # frames as necessary. Assume all audio frames have the same duration.
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


class LanguageDetector(FrameProcessor):
    def __init__(self,context):
        super().__init__()
        self._accumulator = ""
        self._context = context
        self._accumulating_transcript = False
        self.lang_map = {
            lingua.Language.CHINESE: Language.ZH_CN,
            lingua.Language.ENGLISH: Language.EN_US,
            lingua.Language.FRENCH: Language.FR_FR,
            lingua.Language.PORTUGUESE: Language.PT_PT,
            lingua.Language.INDONESIAN: Language.ID,
            lingua.Language.TAGALOG: Language.TL,
            lingua.Language.VIETNAMESE: Language.VI,
            lingua.Language.SPANISH:Language.ES_ES
        }

    def reset(self):
        self._accumulator = ""
        self._accumulating_transcript = False
        self._context.set_messages([])

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._accumulating_transcript = True
        elif isinstance(frame, TextFrame):
            if self._accumulating_transcript:
                self._accumulator += frame.text
            return
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = self._accumulator.strip()
            lang = detector.detect_language_of(text)

            await self.push_frame(TTSUpdateSettingsFrame({
                "language": self.lang_map[lang],
                "voice": language_to_google_tts_language(self.lang_map[lang]) + "-Standard-A"
            }))
            await self.push_frame(TextFrame(text=text))
            self.reset()

        await self.push_frame(frame, direction)


async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport with specific audio parameters
    - Gemini Live multimodal model integration
    - Voice activity detection
    - Animation processing
    - RTVI event handling
    """
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Set up Daily transport with specific audio/video parameters for Gemini
        transport = DailyTransport(
            room_url,
            token,
            "翻译助手",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=False,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001")
    context = OpenAILLMContext([{"role": "system", "content": read_prompt_from_yaml("./config.yaml")}])
    context_aggregator_user = llm.create_context_aggregator(context).user()

    tts = GoogleTTSService(
        voice_id="en-US-Standard-A",
        params=GoogleTTSService.InputParams(language=Language.EN_US),
        credentials=os.getenv("GOOGLE_TEST_CREDENTIALS"),
    )

    audio_collector = UserAudioCollector(context, context_aggregator_user)
    language_detector = LanguageDetector(context)
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,
            audio_collector,
            context_aggregator_user,
            llm,  # LLM
            language_detector,
            tts,  # TTS
            transport.output(),  # Transport bot output
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        await task.queue_frames([context_aggregator_user.get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        # await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
