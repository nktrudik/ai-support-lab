import asyncio

from ai_support_lab.config import Settings
from ai_support_lab.runtime import open_runtime
from ai_support_lab.tickets.schemas import Customer, TicketCreate


async def main() -> None:
    async with open_runtime(Settings()) as runtime:
        for title, description in [
            (
                "Login password rejected",
                "Production outage; all users blocked by authentication failure.",
            ),
            ("Invoice charged twice", "Please check the duplicate subscription payment."),
            ("Request export feature", "Please add an option to export dashboard reports."),
        ]:
            ticket = await runtime.tickets.create(
                TicketCreate(
                    title=title,
                    description=description,
                    customer=Customer(name="Example Customer", tier="pro"),
                    product="cloud",
                )
            )
            print(ticket.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
