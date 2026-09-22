import argparse
import asyncio
import json
import os

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from ai_support_lab.config import Settings


async def main(ticket_id: int) -> None:
    # StdioTransport не наследует произвольные env vars автоматически.
    # Передаём только configuration приложения, чтобы child использовал ту же БД
    # и выбранные providers, не получая остальные секреты окружения родителя.
    environment = {
        name.upper(): os.environ[name.upper()]
        for name in Settings.model_fields
        if name.upper() in os.environ
    }
    transport = StdioTransport(
        command="uv", args=["run", "python", "-m", "ai_support_lab.mcp.server"], env=environment
    )
    async with Client(transport) as client:
        result = await client.call_tool("analyze_ticket", {"ticket_id": ticket_id})
        print(json.dumps(result.structured_content, ensure_ascii=False, indent=2))
        print(await client.read_resource(f"ticket://{ticket_id}"))
        print(await client.get_prompt("support_ticket_analysis", {"ticket_id": str(ticket_id)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticket_id", type=int)
    asyncio.run(main(parser.parse_args().ticket_id))
