import os

from dotenv import load_dotenv
from openai import OpenAI

from .EMB import NVIDIAEmbeddings


load_dotenv()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
CHAT_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"


client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)

embeddings = NVIDIAEmbeddings(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    model_name=EMBED_MODEL,
)


SYSTEM_PROMPT = """
You are a helpful assistant.

Rules:
- Answer the user's questions clearly and naturally.
- When retrieved FAISS context is relevant to professor or institute questions, use it as the primary source.
- If the question is general and not about the indexed professor data, answer from normal knowledge.
- If the question is about a professor or institute detail and the retrieved context does not support the answer, say you could not find that information in the indexed professor data.
- If the user's reference is unclear, use chat history to resolve it. If still unclear, ask for clarification.
- If you are unsure, say so briefly instead of inventing details.
- Keep answers concise unless the user asks for more depth.
""".strip()


def generate_chat_response(user_input: str, context_chunks, chat_history):
    if not NVIDIA_API_KEY:
        return "Server is missing NVIDIA_API_KEY."

    context_text = "\n\n".join(context_chunks).strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in chat_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Relevant context:\n{context_text if context_text else 'No external context provided.'}\n\n"
                f"User question:\n{user_input}"
            ),
        }
    )

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=1,
        top_p=1,
        max_tokens=4096,
        stream=False,
    )

    return (completion.choices[0].message.content or "").strip()
