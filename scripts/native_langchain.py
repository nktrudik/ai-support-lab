import argparse
import asyncio

from ai_support_lab.config import Settings
from ai_support_lab.llm.native_langchain import native_tool_analysis
from ai_support_lab.runtime import open_runtime


async def main(ticket_id: int) -> None:
    settings = Settings()
    async with open_runtime(settings) as runtime:
        result = await native_tool_analysis(settings, runtime.tickets, ticket_id)
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticket_id", type=int)
    asyncio.run(main(parser.parse_args().ticket_id))
