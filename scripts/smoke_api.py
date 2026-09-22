import argparse

import httpx


def main(base_url: str) -> None:
    with httpx.Client(base_url=base_url, timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        created = client.post(
            "/api/v1/tickets",
            json={
                "title": "Login password rejected",
                "description": "Production outage; all users blocked by authentication token failures.",
                "customer": {"name": "Smoke Example", "tier": "enterprise"},
                "product": "cloud",
            },
        )
        created.raise_for_status()
        ticket_id = created.json()["id"]
        for endpoint in ("classify/sklearn", "classify/torch", "analyze"):
            result = client.post(f"/api/v1/tickets/{ticket_id}/{endpoint}", json={})
            result.raise_for_status()
            print(endpoint, result.json())
        print(f"ticket_id={ticket_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    main(parser.parse_args().base_url)
