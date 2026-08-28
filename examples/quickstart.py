"""Minimal end-to-end example against any OpenAI-compatible endpoint.

Run with your endpoint and model:

    python examples/quickstart.py http://127.0.0.1:8000/v1 my-model [api-key]

The extraction batch size is set to 1 here so every exchange consolidates
immediately — good for a demo, wasteful in production (the default of 8
batches turns into one LLM call; see ConsolidationConfig.batch_size).
"""

import asyncio
import sys

from memory_sdk import Memory, MemoryConfig, OpenAICompatibleLLM

EXCHANGES = [
    ("Hi! I'm Aki, I just moved to Osaka for a teaching job.", "Welcome to Osaka, Aki!"),
    ("I adopted a dog last week. His name is Max.", "Max sounds lovely!"),
    ("Max keeps stealing my morning coffee time — I walk him instead.", "A fair trade!"),
]


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    base_url, model = sys.argv[1], sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 else None

    llm = OpenAICompatibleLLM(base_url, model=model, api_key=api_key)
    config = MemoryConfig(
        db_path="quickstart_memory.sqlite3",
        consolidation={"batch_size": 1},  # demo: consolidate every exchange
    )

    async with llm, Memory(config, llm=llm) as mem:
        for user_text, assistant_text in EXCHANGES:
            await mem.add("demo-user", user_text, assistant_text, session_id="demo-chat")
        print("consolidating...")
        await mem.flush(timeout=120.0)

        print("\n--- status ---")
        for key, value in mem.status().items():
            print(f"  {key}: {value}")

        print("\n--- stored memories ---")
        for item in await mem.list("demo-user"):
            print(f"  [{item.mem_type}] ({item.subject}) {item.content}")

        print("\n--- recall: 'what pet does the user have?' ---")
        result = await mem.search("demo-user", "what pet does the user have?")
        print(mem.render(result))

        print("\n--- core memory (pinned) ---")
        print(await mem.render_core("demo-user") or "(empty)")


if __name__ == "__main__":
    asyncio.run(main())
