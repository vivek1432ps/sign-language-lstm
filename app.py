"""Serve SignMotion Lab on loopback. The demo only reads the pinned UCI dataset."""
from html import escape
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from sequence import CLASSES, HELD, ROOT, CACHE, SignLSTM, load_archive, predict

CSS = """
:root{color-scheme:dark;--fg:#fcf6ec;--muted:#b5a8b9;--lilac:#d2adff;--lime:#d8fa7a;--edge:#4a394d;--base:#16111a}
*{box-sizing:border-box}body{background:radial-gradient(ellipse at 8% 0%,#302139 0,transparent 35%),var(--base);
color:var(--fg);margin:0;font:15px/1.5 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:0 28px 40px}
h1,h2,p{margin:0}h1{font-size:clamp(35px,5.1vw,67px);line-height:1.05;letter-spacing:-.06em}h1 em{color:var(--lilac);font-style:normal}
h2{font-size:21px;letter-spacing:-.025em;margin-bottom:9px}.muted,small{color:var(--muted)}
header{display:flex;justify-content:space-between;align-items:center;padding:20px 0;border-bottom:1px solid var(--edge);gap:12px}
.brand,.tag,.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;font-weight:780}
.brand{color:var(--lilac)}.tag{border:1px solid #5c4a61;padding:5px 11px;border-radius:30px}
.hero{padding:24px 0 19px}.hero p{margin-top:8px}.eyebrow{color:var(--lilac);margin-bottom:5px}
.deck{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:17px}
.card{min-width:0;background:#241a29;border:1px solid var(--edge);border-radius:19px;padding:21px}
.result{background:radial-gradient(ellipse at 90% 0%,#362340 0,transparent 54%),#241a29}
form{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:end;gap:10px;margin:11px 0}
label{display:block;font-weight:700;margin-bottom:5px}select,button{font:inherit;border-radius:8px;padding:9px 10px}
select{width:100%;background:#170f1b;color:var(--fg);border:1px solid #65506d}
button{background:var(--lime);color:#252614;border:0;cursor:pointer;font-weight:800}
:focus-visible{outline:2px solid var(--lilac);outline-offset:3px}
.figure{margin:14px 0 4px;background:#18121d;border:1px solid var(--edge);border-radius:11px;padding:8px}
.figure svg{display:block;width:100%;height:auto}.keys{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px}
.keys i{display:inline-block;width:9px;height:9px;border-radius:9px;margin-right:5px}
.guess{font-size:clamp(29px,3.8vw,51px);line-height:1.08;font-weight:825;letter-spacing:-.05em;color:var(--lime);margin:3px 0 10px;text-transform:capitalize}
.scores{list-style:none;margin:10px 0 8px;padding:0}
.scores li{display:grid;align-items:center;grid-template-columns:74px 1fr 46px;gap:9px;margin:7px 0;font-size:13px}
.track{height:7px;background:#49384e;border-radius:10px}.fill{height:7px;border-radius:10px;background:var(--lilac)}
.metrics{margin-top:15px}.numbers{display:flex;gap:14px 26px;flex-wrap:wrap}.numbers b{color:var(--lime);font-size:19px;margin-right:4px}
details{margin-top:10px;color:var(--muted)}summary{cursor:pointer;color:var(--fg)}a{color:var(--lilac)}.fault{color:#ffc2cc}
table{border-collapse:collapse;margin:8px 0;text-align:right;width:100%}th,td{padding:5px 6px;border-bottom:1px solid var(--edge)}
@media(max-width:740px){main{padding-inline:15px}.deck{grid-template-columns:1fr}.card{padding:18px}header{gap:3px}.tag{font-size:9px}}
"""

def gloss(label):
    return "wait (not yet)" if label == "wait(notyet)" else label


def graph(x):
    colors = ("#d5abff","#dafb85","#ffb69d")
    lines = []
    # Common extent across x/y/z for this clip. Absolute axes are glove coordinates.
    low, high = min(-.2, x[:,:3].min()), max(.2, x[:,:3].max())
    for i, color in enumerate(colors):
        points = " ".join(f"{20+j*420/(len(x)-1):.1f},{127-(v-low)/(high-low)*102:.1f}" for j,v in enumerate(x[:,i]))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
    grid = "".join(f'<path d="M20,{i} H440" stroke="#48394e"/>' for i in (25,59,93,127))
    keys = "".join(f'<span><i style="background:{c}"></i>{k} coordinate</span>' for k,c in zip("XYZ",colors))
    return f'<figure class="figure"><svg viewBox="0 0 460 153" role="img" aria-label="Glove x, y, z traces over 48 normalized steps">{grid}{"".join(lines)}<text x="20" y="146" fill="#b5a8b9" font-size="11">START</text><text x="440" y="146" text-anchor="end" fill="#b5a8b9" font-size="11">END</text></svg><figcaption class="keys">{keys}</figcaption></figure>'


def load_bundle(archive=CACHE):
    torch.set_num_threads(2)
    checkpoint = torch.load(ROOT/"model.pt", weights_only=True, map_location="cpu")
    model = SignLSTM()
    model.load_state_dict(checkpoint["state"])
    model.eval()
    x, y, groups, _ = load_archive(archive)
    subset = [(x[i], int(y[i])) for i,p in enumerate(groups) if p == HELD]
    xs = [elt[0] for elt in subset]
    prob = predict(model, np.stack(xs), checkpoint["mean"].numpy(), checkpoint["scale"].numpy())
    report = json.loads((ROOT/"report.json").read_text(encoding="utf-8"))
    return subset, prob, report


def render(bundle, index, error=None):
    subset, probability, report = bundle
    x, expected = subset[index]
    p = probability[index]
    ranks = p.argsort()[::-1]
    options = "".join(f'<option value="{i}" {"selected" if i == index else ""}>Clip {i+1:03d}</option>' for i in range(len(subset)))
    top = int(ranks[0])
    bars = "".join(f'<li><span>{gloss(CLASSES[i])}</span><div class="track"><div class="fill" style="width:{p[i]*100:.3f}%"></div></div><b>{p[i]:.0%}</b></li>' for i in ranks[:5])
    rows = ''.join(f"<tr><td>{f['held_out']}</td><td>{f['test']}</td><td>{f['lstm']['accuracy']:.1%}</td><td>{f['baseline']['accuracy']:.1%}</td></tr>" for f in report["folds"])
    err = f'<p role="alert" class="fault">{escape(error)}</p>' if error else ''
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SignMotion Lab | Auslan recognition</title><style>{CSS}</style><body><main>
<header><div class="brand">SIGNMOTION / LAB</div><div class="tag">Auslan · 10 isolated signs</div></header>
<section class="hero"><div class="eyebrow">Sequence recognition / CPU demo</div>
<h1>From motion <em>to a sign.</em></h1><p class="muted">Explore what a trained LSTM sees in recorded Australian Sign Language glove signals.</p></section>
<div class="deck">
<section class="card"><h2>Recorded input</h2><p class="muted">Choose a clip from a signer the demo model never saw during training.</p>
{err}<form method="get" action="/"><div><label for="clip">Held-out recording</label><select id="clip" name="clip">{options}</select></div><button>Classify clip →</button></form>
{graph(x)}<small>Showing 3 of 9 features. Each recording is resampled to 48 steps.</small></section>
<section class="card result"><div class="eyebrow">LSTM prediction / clip {index+1:03d}</div><div class="guess">{gloss(CLASSES[top])}</div>
<p>Ground truth: <strong>{gloss(CLASSES[expected])}</strong> · {"Correct" if top==expected else "Incorrect"}</p>
<ol aria-label="Top five model scores" class="scores">{bars}</ol>
<small>Softmax scores rank the known labels. They are not calibrated confidence and cannot reject unknown signs.</small></section>
</div>
<section class="card metrics"><div class="eyebrow">Evaluation / five held-out signers</div>
<div class="numbers"><span><b>{report['lstm']['accuracy']:.1%}</b>LSTM accuracy</span><span><b>{report['baseline']['accuracy']:.1%}</b>baseline accuracy</span><span><b>{report['lstm']['macro_f1']:.3f}</b>LSTM macro-F1</span><span><b>{report['total']}</b>test clips</span></div>
<details><summary>Results by signer and dataset notes</summary>
<table><thead><tr><th>Signer</th><th>Clips</th><th>LSTM</th><th>Baseline</th></tr></thead><tbody>{rows}</tbody></table>
<p>Source: <a href="https://doi.org/10.24432/C5XG6C">M. W. Kadous, Australian Sign Language Signs (UCI)</a>. This is a 10-class benchmark on recordings from five signers using an instrumented glove. It does not recognize webcam input, continuous signing, ASL or Indian Sign Language, or translate sentences.</p></details>
</section></main></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlparse(self.path)
        if route.path != "/":
            self.send_error(404)
            return
        try:
            clip = int(parse_qs(route.query, max_num_fields=8).get("clip",["0"])[0])
            if not 0 <= clip < len(self.server.bundle[0]):
                raise ValueError("unknown clip")
            status, display = 200, render(self.server.bundle,clip)
        except ValueError:
            status, display = 400, render(self.server.bundle,0,"Choose a recording from the clip list.")
        body = display.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--archive", type=Path, default=CACHE)
    args = parser.parse_args()
    bundle = load_bundle(args.archive)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.bundle = bundle
        print(f"Open http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()
