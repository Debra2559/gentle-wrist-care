const path = require("path");
const puppeteer = require(path.join(__dirname, "..", "..", "node_modules", "puppeteer"));

(async () => {
  const htmlPath = path.join(__dirname, "calibration_state_machine.html");
  const outPath = path.join(__dirname, "calibration_state_machine.png");

  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--font-render-hinting=none"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 2 });
  await page.goto("file://" + htmlPath, { waitUntil: "networkidle0" });
  await page.waitForFunction("window.__diagramReady === true", { timeout: 60000 });

  await page.evaluate(() => {
    const svg = document.querySelector("#mount svg");
    const canvas = document.getElementById("canvas");
    const bbox = svg.getBBox();
    const w = bbox.width + 40;
    const h = bbox.height + 40;
    svg.setAttribute("viewBox", `${bbox.x - 20} ${bbox.y - 20} ${w} ${h}`);
    svg.removeAttribute("style");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    const availW = canvas.clientWidth;
    const availH = canvas.clientHeight;
    const scale = Math.min(availW / w, availH / h);
    svg.setAttribute("width", String(Math.floor(w * scale)));
    svg.setAttribute("height", String(Math.floor(h * scale)));
  });

  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: outPath, type: "png" });
  await browser.close();
  console.log("saved: " + outPath);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
