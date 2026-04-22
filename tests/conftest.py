"""Pytest local configuration: minimal env for importing ``ssd`` without a full setup."""

import os

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
