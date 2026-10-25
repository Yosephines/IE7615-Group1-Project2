"""Split reports/baseline_comparisons.png into one JPEG per prompt and insert the gallery into the baseline memo source."""
from pathlib import Path
import csv, html, re
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/src/figures/baseline'
MEMO = ROOT / 'reports/src/milestone01_baseline_gallery_memo.md'
Image.MAX_IMAGE_PIXELS = None

def license_label(url):
    # e.g. http://creativecommons.org/licenses/by-sa/2.0/ -> CC BY-SA 2.0
    kind, version = url.rstrip('/').split('/')[-2:]
    return f'CC {kind.upper()} {version}'

def main():
    prompts = list(csv.DictReader((ROOT / 'data/baseline_prompts.csv').open(encoding='utf-8')))
    manifest = {r['pair_id']: r for r in csv.DictReader((ROOT / 'data/manifest.csv').open(encoding='utf-8'))}
    sheet = Image.open(ROOT / 'reports/baseline_comparisons.png').convert('RGB')
    # The sheet stacks the 16 prompts as equal-height rows (original | generated).
    OUT.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, row in enumerate(prompts):
        top, bottom = round(i * sheet.height / len(prompts)), round((i + 1) * sheet.height / len(prompts))
        strip = sheet.crop((0, top, sheet.width, bottom))
        strip.thumbnail((900, 900), Image.Resampling.LANCZOS)
        strip.save(OUT / f"{row['prompt_id']}.jpg", quality=85)
        src = manifest[row['pair_id']]
        cells.append(
            f'<figure><img src="figures/baseline/{row["prompt_id"]}.jpg" alt="{row["prompt_id"]}">'
            f'<figcaption><b>{row["prompt_id"]}</b> {html.escape(row["prompt"])} '
            f'<span class="credit">Original: COCO {src["image_id"]}, <a href="{src["source_url"]}">Flickr</a>, {license_label(src["license_url"])}. Seed {row["seed"]}.</span>'
            f'</figcaption></figure>')
    block = '<!-- gallery:start -->\n<div class="gallery">\n' + '\n'.join(cells) + '\n</div>\n<!-- gallery:end -->'
    memo = MEMO.read_text(encoding='utf-8')
    MEMO.write_text(re.sub(r'<!-- gallery:start -->.*?<!-- gallery:end -->', lambda _: block, memo, flags=re.S), encoding='utf-8')
    print(f'Wrote {len(cells)} gallery strips to {OUT.relative_to(ROOT)}')

if __name__ == '__main__':
    main()
