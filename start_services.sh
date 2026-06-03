#!/bin/bash
cd /home/lance/TheFoolsGambitPython
nohup python3 train_server.py > /tmp/training.log 2>&1 &
nohup /home/lance/.local/bin/cloudflared --config /home/lance/.cloudflared/config.yml tunnel run > /tmp/tunnel.log 2>&1 &
