# ptp_config_assistant.py

"""
This script uses sentence-transformers to vectorize PTP config files and match user queries
to the most relevant configuration. If modification is requested, it uses a local LLM (via Ollama)
to rewrite the YAML config accordingly.
"""

import os
import torch
import subprocess
from sentence_transformers import SentenceTransformer, util

# 1. Load sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load and embed PTP config
with open("PtpConfigBoundary.yaml", "r") as f:
    ptp_config_text = f.read()

config_embedding = model.encode(ptp_config_text, convert_to_tensor=True)

# 3. Accept a user query
query = input("Ask something like 'ptp config for dual nic': ")
query_embedding = model.encode(query, convert_to_tensor=True)

# 4. Compute similarity
similarity = util.pytorch_cos_sim(query_embedding, config_embedding).item()
print(f"\n[Similarity Score] {similarity:.2f}")

if similarity > 0.4:
    print("\n✅ Closest PTP config found:\n")
    print(ptp_config_text[:1000])  # Preview first part of config

    # 5. Optional: Modify Config using Ollama
    modify = input("\nDo you want to change anything in the config? (e.g., interface name): ")
    if modify.strip():
        print("\n🔧 Calling local LLM (Ollama) to modify configuration...")

        prompt = f"""
### SYSTEM:
You are a YAML configuration assistant. Modify the following PTP config based on the user's instruction.

### CONFIG:
{ptp_config_text}

### USER REQUEST:
{modify}

### NEW CONFIG:
"""

        def call_ollama(prompt, model="mistral"):
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode(),
                capture_output=True
            )
            return result.stdout.decode()

        updated_config = call_ollama(prompt)
        print("\n📄 Modified Config Snippet:\n")
        print(updated_config.strip())
else:
    print("\n❌ No strong config match found.")
