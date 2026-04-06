from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "screenshots"
BASE_URL = "http://127.0.0.1:5000"

FORM_DATA = {
    "Temperature": "20",
    "RH": "35",
    "Ws": "10",
    "Rain": "0",
    "FFMC": "85",
    "DMC": "10",
    "ISI": "5",
    "BUI": "12",
    "Region_encoded": "1",
}


def fill_and_submit(page) -> None:
    for field, value in FORM_DATA.items():
        page.fill(f'input[name="{field}"]', value)
    page.click('button[type="submit"]')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})

        page.goto(BASE_URL, wait_until="networkidle")
        page.screenshot(path=str(OUT_DIR / "01-home-input.png"), full_page=True)

        fill_and_submit(page)
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(OUT_DIR / "02-prediction-output.png"), full_page=True)

        browser.close()

    print("Screenshots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
