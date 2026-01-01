#!/bin/bash
MODEL_DIR="pretrained_models/Fun-CosyVoice3-0.5B"
uv run python openai_server.py --port 50000 --model_dir "$MODEL_DIR"
