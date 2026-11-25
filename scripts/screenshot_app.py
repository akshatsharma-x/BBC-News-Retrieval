"""Start the Streamlit app headlessly and capture interface screenshots with Chrome.

Writes outputs/figures/ui_search.png, ui_compare.png, ui_benchmark.png and ui_explorer.png.
Usage:  python scripts/screenshot_app.py
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
PORT = 8599
WIDTH = 1440


def wait_for_server(url: str, timeout: float = 180) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)


def clip_between(page, top_sel: str, bottom_sel: str, pad_top: int = 20, pad_bottom: int = 24) -> dict:
    """Clip rectangle spanning from the first top_sel element to the last bottom_sel element."""
    top = page.locator(top_sel).first.bounding_box()
    bottoms = [b for b in (page.locator(bottom_sel).nth(i).bounding_box()
                           for i in range(page.locator(bottom_sel).count())) if b]
    y0 = max(0, top["y"] - pad_top)
    y1 = max(b["y"] + b["height"] for b in bottoms) + pad_bottom
    return {"x": 150, "y": y0, "width": WIDTH - 300, "height": y1 - y0}


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server(f"http://localhost:{PORT}/_stcore/health")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": WIDTH, "height": 2600}, device_scale_factor=2)
            page.goto(f"http://localhost:{PORT}", wait_until="networkidle")
            page.wait_for_selector(".card", timeout=240_000)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(FIG / "ui_search.png"), clip={"x": 150, "y": 0, "width": WIDTH - 300, "height": 1240})

            page.get_by_role("tab", name="Compare models").click()
            page.wait_for_selector(".deskcol", timeout=60_000)
            page.wait_for_timeout(1000)
            page.screenshot(path=str(FIG / "ui_compare.png"), clip=clip_between(page, "[data-testid='stTabs']", ".deskcol", 4))

            page.get_by_role("tab", name="Benchmark").click()
            page.wait_for_selector(".kpis", timeout=60_000)
            page.wait_for_function("() => document.querySelectorAll('.vega-embed').length >= 3", timeout=60_000)
            page.wait_for_selector(".judged", timeout=60_000)
            page.wait_for_timeout(2500)
            page.screenshot(path=str(FIG / "ui_benchmark.png"),
                            clip=clip_between(page, ".kpis", "[data-testid='stVegaLiteChart']"))
            page.screenshot(path=str(FIG / "ui_explorer.png"), clip=clip_between(page, ".topic", ".judged", 108))
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    print("saved ui_search.png, ui_compare.png, ui_benchmark.png, ui_explorer.png to", FIG)


if __name__ == "__main__":
    main()
