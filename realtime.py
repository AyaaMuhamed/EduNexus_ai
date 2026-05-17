import asyncio
from recorder import record_chunk
from Buffering import BufferManager
from transcriber import transcribe_audio

async def run_realtime(verbose=True):
    print("🚀 REALTIME TRANSCRIPTION STARTED")

    buffer = BufferManager(max_chunks=15, verbose=verbose)

    while True:
        # 1 second audio capture
        audio = record_chunk()

        if verbose:
            print("[RECORDER] captured 1s audio")

        buffer.add(audio)

        # when enough audio collected
        if buffer.is_ready():
            print("\n🚀 [PIPELINE] Sending request...\n")

            audio_data = buffer.get_audio()

            text = await transcribe_audio(audio_data)

            print("📝 [TRANSCRIPT]:", text)
            print("--------------------------------------------------")