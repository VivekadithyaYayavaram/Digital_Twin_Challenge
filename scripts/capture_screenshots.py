"""Capture real dashboard screenshots for the slides / README.

Usage (needs `pip install playwright && python -m playwright install chromium`):
    streamlit run app.py --server.port 8599 &      # start the app first
    python scripts/capture_screenshots.py --url http://localhost:8599
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "submission" / "screenshots"
TABS = [("Overview", "01_overview"), ("Patient timeline", "02_timeline"),
        ("Two-stream fusion", "03_fusion"), ("Prediction & explanation", "04_prediction"),
        ("Digital Twin simulation", "05_simulation"), ("Model evaluation", "06_evaluation"),
        ("Responsible use", "07_responsible_use")]


def wait_idle(page, extra=1.5):
    page.wait_for_selector('[data-testid="stApp"]')
    for _ in range(120):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.25)
    time.sleep(extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8599")
    ap.add_argument("--advance", type=int, default=3, help="number of '+2 h' clicks before capture")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.5)
        page.goto(a.url)
        wait_idle(page, 4)
        for _ in range(a.advance):
            page.get_by_role("button", name="⏩ +2 h").click()
            wait_idle(page)
        for tab, name in TABS:
            page.get_by_role("tab", name=tab).click()
            wait_idle(page, 2.5)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            print("saved", name)
        b.close()


if __name__ == "__main__":
    main()
