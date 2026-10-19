"""
Render docs/demo.gif from a real run of demo.py, so the GIF cannot drift from the code.

    uv run --no-project --with pillow --with-requirements requirements.txt python docs/make_demo_gif.py

demo.py runs in a pseudo-terminal with --replay, so the colours and pauses are
the ones a viewer would see. Before writing, the GIF's text is checked line for
line against a plain run of demo.py.
"""

import os
import pty
import re
import select
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo.gif"
COMMAND = "python demo.py"
ROWS = 30
FONT_FILE, FONT_SIZE = "/System/Library/Fonts/Menlo.ttc", 15
PAD, BAR = 18, 30

BACKGROUND, BAR_COLOUR, DEFAULT = "#1b1f27", "#2a2f3a", "#d8dee9"
COLOURS = {"31": "#ff6b6b", "32": "#7ee787", "33": "#f2cc60", "36": "#79c0ff"}
DIM, BOLD = "#7b8394", "#ffffff"
SGR = re.compile(r"\x1b\[([0-9;]*)m")


def capture():
    """Run demo.py in a pseudo-terminal; return (seconds, line) as each line appears."""
    master, slave = pty.openpty()
    env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
    proc = subprocess.Popen([sys.executable, "demo.py", "--replay"], cwd=ROOT,
                            stdin=slave, stdout=slave, stderr=slave, env=env)
    os.close(slave)
    events, pending, start = [], b"", time.monotonic()
    while True:
        ready, _, _ = select.select([master], [], [], 0.1)
        if not ready:
            if proc.poll() is not None:
                break
            continue
        try:
            chunk = os.read(master, 4096)
        except OSError:  # macOS raises EIO once the child has exited
            break
        if not chunk:
            break
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            events.append((time.monotonic() - start, line.decode().rstrip("\r")))
    proc.wait()
    os.close(master)
    if pending.strip():
        events.append((time.monotonic() - start, pending.decode().rstrip("\r")))
    return events


def spans(line):
    """Split an ANSI-coloured line into (text, colour, bold) runs."""
    out, colour, bold, dim, pos = [], None, False, False, 0
    for match in SGR.finditer(line):
        if match.start() > pos:
            out.append((line[pos:match.start()], colour, bold, dim))
        for code in (match.group(1) or "0").split(";"):
            if code == "0":
                colour, bold, dim = None, False, False
            elif code == "1":
                bold = True
            elif code == "2":
                dim = True
            elif code in COLOURS:
                colour = COLOURS[code]
        pos = match.end()
    out.append((line[pos:], colour, bold, dim))
    return [(text, c or (DIM if d else BOLD if b else DEFAULT), b) for text, c, b, d in out if text]


def strip(line):
    return SGR.sub("", line)


class Screen:
    def __init__(self, width_cols):
        self.regular = ImageFont.truetype(FONT_FILE, FONT_SIZE, index=0)
        self.bold = ImageFont.truetype(FONT_FILE, FONT_SIZE, index=1)
        self.char_w = self.regular.getlength("M")
        self.line_h = int(FONT_SIZE * 1.4)
        self.size = (int(PAD * 2 + self.char_w * width_cols), BAR + PAD * 2 + self.line_h * ROWS)

    def render(self, lines):
        image = Image.new("RGB", self.size, BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, self.size[0], BAR], fill=BAR_COLOUR)
        for i, dot in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            draw.ellipse([14 + i * 20, 10, 26 + i * 20, 22], fill=dot)
        for row, line in enumerate(lines[-ROWS:]):
            x, y = PAD, BAR + PAD + row * self.line_h
            for text, colour, bold in spans(line):
                font = self.bold if bold else self.regular
                draw.text((x, y), text, font=font, fill=colour)
                x += self.char_w * len(text)
        return image


def main():
    events = capture()
    plain = subprocess.run([sys.executable, "demo.py"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.splitlines()
    shown = [strip(line) for _, line in events]
    if [l.rstrip() for l in shown] != [l.rstrip() for l in plain]:
        print("GIF text does not match a plain run of demo.py; not writing it.")
        return 1

    prompt = "\x1b[32m$\x1b[0m "
    screen = Screen(max([len(l) for l in plain] + [80]) + 2)
    frames, durations = [], []

    def frame(lines, ms):
        frames.append(screen.render(lines))
        durations.append(int(ms))

    frame([prompt], 600)
    for n in range(1, len(COMMAND) + 1):
        frame([prompt + COMMAND[:n]], 45)
    frame([prompt + COMMAND], 500)

    lines = [prompt + COMMAND]
    i = 0
    while i < len(events):
        # Lines printed together (no pause between them) become one frame.
        j = i
        while j + 1 < len(events) and events[j + 1][0] - events[j][0] < 0.05:
            j += 1
        lines += [line for _, line in events[i:j + 1]]
        wait = events[j + 1][0] - events[j][0] if j + 1 < len(events) else 5.0
        frame(lines, min(max(wait, 0.05), 3.0) * 1000 if j + 1 < len(events) else 5000)
        i = j + 1

    palette = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    palette[0].save(OUT, save_all=True, append_images=palette[1:], duration=durations,
                    loop=0, optimize=True, disposal=1)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(frames)} frames, "
          f"{sum(durations) / 1000:.1f}s, {OUT.stat().st_size // 1024} KB, "
          f"text verified against a plain run ({len(plain)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
