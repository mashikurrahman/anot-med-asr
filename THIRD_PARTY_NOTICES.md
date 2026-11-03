# Third-party notices

The Anot Medical ASR engine is built on the open-source components below.

| Component | Used for | License |
|---|---|---|
| [Whisper](https://github.com/openai/whisper) (OpenAI), model `large-v3` | Speech recognition. The engine serves a fine-tuned version of it. | MIT |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (SYSTRAN) | Whisper inference | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) (OpenNMT) | Fast GPU/CPU inference engine | MIT |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) (Hervé Bredin and contributors) | Speaker diarization code | MIT |
| [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) (pyannoteAI) | Speaker diarization model weights ("Speaker 1", "Speaker 2" labels) | CC-BY-4.0 |
| [PyTorch](https://pytorch.org) | Runs the speaker diarization model | BSD-3-Clause |

## Attribution required by CC-BY-4.0

Speaker labels are produced with the **pyannote/speaker-diarization-community-1** model by
**pyannoteAI**, licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The model is used as published, without modification.
