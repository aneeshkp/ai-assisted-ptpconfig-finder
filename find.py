# ptp_config_assistant.py

"""
This script uses sentence-transformers to vectorize PTP config files and match user queries
to the most relevant configuration. If modification is requested, it uses a local LLM (like Mistral)
to rewrite the YAML config accordingly.
"""

import os
import torch
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama

# 1. Load sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load and embed PTP config
with open("PtpConfigBoundary.yaml", "r") as f:
    ptp_config_text = f.read()

config_embedding = model.encode(ptp_config_text, convert_to_tensor=True)

# 3. Accept a user query
query = input("Ask something like 'ptp config for boundary nic': ")
query_embedding = model.encode(query, convert_to_tensor=True)

# 4. Compute similarity
similarity = util.pytorch_cos_sim(query_embedding, config_embedding).item()
print(f"\n[Similarity Score] {similarity:.2f}")

if similarity > 0.4:
    print("\n✅ Closest PTP config found:\n")
    print(ptp_config_text[:1000])  # Preview first part of config

    # 5. Optional: Modify Config with LLM
    modify = input("\nDo you want to change anything in the config? (e.g., interface name): ")
    if modify.strip():
        print("\n🔧 Calling local LLM to modify configuration...")

        # Load local model (update path to your GGUF model)
        llm = Llama(model_path="./models/mistral-7b-instruct.Q4_K_M.gguf", n_ctx=2048)

        prompt = f"""
### SYSTEM:
You are a YAML configuration assistant. Modify the following PTP config based on the user's instruction.

### CONFIG:
{ptp_config_text}

### USER REQUEST:
{modify}

### NEW CONFIG:
"""

        response = llm(prompt, max_tokens=1024, stop=["###"], echo=False)
        print("\n📄 Modified Config Snippet:\n")
        print(response["choices"][0]["text"].strip())
else:
    print("\n❌ No strong config match found.")
