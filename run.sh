#!/bin/bash
# conda activate conda_env/
/home/zadmin/whisper_streaming/conda_env/bin/python3 -m uvicorn whisper_streaming:app --reload --host localhost --port 8003