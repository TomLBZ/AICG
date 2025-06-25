#!/bin/sh

curl https://ai.stdev.remoteblossom.com/engines/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ai/phi4",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Please write a shortest echo function in python. Show the code only."
            }
        ]
    }'