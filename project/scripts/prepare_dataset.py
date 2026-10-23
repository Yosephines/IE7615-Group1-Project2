"""Prepare a fixed COCO caption subset and record the choices needed to rebuild it."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, io, json, random, re, unicodedata, zipfile
import requests
import numpy as np
from PIL import Image, ImageOps
ROOT=Path(__file__).resolve().parents[2]
SEED=42
SIZE=1000
RAW=ROOT/'data/raw'
OUT=ROOT/'data/processed/images'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def clean_caption(value):
    value=unicodedata.normalize('NFKC',value)
    value=re.sub(r'\s+',' ',value).strip()
    words=value.split()
    if not 5<=len(words)<=35 or re.search(r'https?://|www\.',value,re.I): return None
    if sum(c.isalpha() for c in value)/max(len(value),1)<0.5:return None
    return value

def annotations():
    RAW.mkdir(parents=True,exist_ok=True)
    archive=RAW/'annotations_trainval2017.zip'
    names=['captions_val2017.json','instances_val2017.json']
    if not all((RAW/n).exists() for n in names):
        if not archive.exists():
            with requests.get('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',stream=True,timeout=(20,90)) as r:
                r.raise_for_status()
                part=archive.with_suffix('.part')
                with part.open('wb') as f:
                    for chunk in r.iter_content(1024*1024):f.write(chunk)
                part.replace(archive)
        with zipfile.ZipFile(archive) as z:
            for name in names:(RAW/name).write_bytes(z.read('annotations/'+name))
    return [json.loads((RAW/n).read_text()) for n in names]

def download(meta):
    dest=RAW/'images'/meta['file_name'];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        url='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+meta['file_name']
        for attempt in range(3):
            try:
                r=requests.get(url,timeout=(20,60));r.raise_for_status()
                dest.with_suffix('.part').write_bytes(r.content)
                dest.with_suffix('.part').replace(dest)
                break
            except requests.RequestException:
                if attempt==2:raise
    return dest

def prepare():
    cap,inst=annotations();OUT.mkdir(parents=True,exist_ok=True)
    by_image=defaultdict(list)
    for ann in sorted(cap['annotations'],key=lambda a:a['id']):by_image[ann['image_id']].append(ann)
    categories={c['id']:c['name'] for c in inst['categories']}
    image_categories=defaultdict(set)
    for ann in inst['annotations']:image_categories[ann['image_id']].add(categories[ann['category_id']])
    licenses={l['id']:l for l in cap['licenses']}
    overrides=json.loads((ROOT/'project/configs/caption_overrides.json').read_text())
    candidates=[];seen_captions=set();counts=Counter()
    images=sorted(cap['images'],key=lambda m:m['id']);random.Random(SEED).shuffle(images)
    for meta in images:
        if meta['license'] not in (4, 5):
            counts['excluded_image_licenses']+=1
            continue
        if overrides.get(str(meta['id']),{}).get('exclude'):
            counts['visual_review_image_exclusions']+=1
            continue
        selected=None
        choices=by_image[meta['id']]
        override=overrides.get(str(meta['id']))
        if override:
            assert any(a['id']==override['caption_id'] for a in choices), 'Caption override must belong to its image'
            choices=sorted(choices,key=lambda a:(a['id']!=override['caption_id'],a['id']))
        for ann in choices:
            value=override.get('caption',ann['caption']) if override and ann['id']==override['caption_id'] else ann['caption']
            caption=clean_caption(value)
            if caption is None:counts['caption_filter_rejections']+=1;continue
            key=caption.casefold()
            if key in seen_captions:counts['duplicate_caption_rejections']+=1;continue
            selected=(ann,caption,key);break
        if selected:
            ann,caption,key=selected;seen_captions.add(key)
            candidates.append((meta,ann,caption))
        else:counts['images_without_eligible_caption']+=1
    rows=[];seen_images=set();seen_pixels=set()
    for start in range(0,len(candidates),128):
        chunk=candidates[start:start+128]
        with ThreadPoolExecutor(max_workers=10) as pool:paths=list(pool.map(download,[a[0] for a in chunk]))
        for (meta,ann,caption),path in zip(chunk,paths):
            try:
                with Image.open(path) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');w,h=im.size
                    if min(w,h)<128:counts['small_image_rejections']+=1;continue
                    pixel_hash=hashlib.sha256(f'{w}x{h}:'.encode()+im.tobytes()).hexdigest()
                    if pixel_hash in seen_images:counts['duplicate_source_image_rejections']+=1;continue
                    # Padding retained the full scene and avoided stretching the subjects.
                    processed=ImageOps.pad(im,(512,512),method=Image.Resampling.LANCZOS,color=(127,127,127))
                    processed_hash=hashlib.sha256(processed.tobytes()).hexdigest()
                    if processed_hash in seen_pixels:counts['duplicate_processed_image_rejections']+=1;continue
                    target=OUT/(str(meta['id']).zfill(12)+'.png')
                    unchanged=False
                    if target.exists():
                        try:
                            with Image.open(target) as saved:
                                unchanged=saved.mode=='RGB' and saved.size==(512,512) and saved.tobytes()==processed.tobytes()
                        except OSError:
                            pass
                    if not unchanged:processed.save(target)
            except (OSError,ValueError):counts['invalid_image_rejections']+=1;continue
            seen_images.add(pixel_hash);seen_pixels.add(processed_hash)
            lic=licenses[meta['license']]
            rows.append(dict(pair_id=f"coco2017val_{meta['id']:012d}",image_id=meta['id'],caption_id=ann['id'],image_path=target.relative_to(ROOT).as_posix(),caption=caption,original_caption=ann['caption'],caption_origin=('reviewed_correction' if overrides.get(str(meta['id']),{}).get('caption') else 'COCO annotation'),source_width=w,source_height=h,image_sha256=digest(target),source_pixel_sha256=pixel_hash,license_name=lic['name'],license_url=lic['url'],source_url=meta.get('flickr_url',''),download_url='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+meta['file_name'],categories=';'.join(sorted(image_categories[meta['id']]))))
            if len(rows)==SIZE:break
        print(f'Prepared {len(rows)}/{SIZE} pairs',flush=True)
        if len(rows)==SIZE:break
    if len(rows)!=SIZE:raise RuntimeError(f'Only {len(rows)} eligible unique pairs')
    # Review corrections kept existing split assignments fixed.
    split_path=ROOT/'project/configs/splits.json'
    split=json.loads(split_path.read_text()) if split_path.exists() else {}
    split={row['pair_id']:split[row['pair_id']] for row in rows if row['pair_id'] in split}
    targets={'development':800,'validation':100,'evaluation':100}
    actual=Counter(split.values())
    missing=[row['pair_id'] for row in rows if row['pair_id'] not in split]
    random.Random(SEED+1).shuffle(missing)
    for name,target in targets.items():
        for _ in range(target-actual[name]):split[missing.pop()]=name
    assert not missing and len(split)==SIZE
    split_path.write_text(json.dumps(split,indent=2)+'\n')
    for row in rows:row['split']=split[row['pair_id']]
    def write_csv(path,values):
        with path.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write_csv(ROOT/'data/manifest.csv',rows)
    remaining=[r for r in rows if r['split']=='evaluation'];prompts=[];covered=set()
    for i in range(16):
        choice=max(remaining,key=lambda r:(len(set(r['categories'].split(';'))-covered),-int(r['image_id'])))
        remaining.remove(choice);covered.update(choice['categories'].split(';'))
        prompts.append(dict(prompt_id=f'p{i+1:02d}',pair_id=choice['pair_id'],prompt=choice['caption'],ground_truth_path=choice['image_path'],seed=SEED+i,categories=choice['categories']))
    write_csv(ROOT/'data/baseline_prompts.csv',prompts)
    category_counts=Counter(c for r in rows for c in r['categories'].split(';') if c)
    summary=dict(dataset='COCO 2017 validation captions',seed=SEED,pairs=len(rows),unique_images=len(seen_images),reviewed_caption_selections=sum(not v.get('exclude') for v in overrides.values()),reviewed_caption_corrections=sum(bool(v.get('caption')) for v in overrides.values()),reviewed_image_exclusions=sum(bool(v.get('exclude')) for v in overrides.values()),allowed_image_license_ids=[4,5],source_images=len(images),source_captions=len(cap['annotations']),eligible_candidate_images=len(candidates),caption_rules='NFKC; whitespace; 5-35 words; no URLs; >=50% alphabetic characters; case-insensitive exact deduplication',image_rules='EXIF transpose; RGB; minimum side 128; decoded exact-image deduplication; aspect-preserving pad to 512x512; PNG',filter_scope='License counts cover all 5000 source images; caption counts cover license-eligible images until the first eligible caption per image; image rejections cover candidates examined until the target was reached.',rejections=dict(counts),split_counts=dict(Counter(r['split'] for r in rows)),word_counts=dict(min=min(len(r['caption'].split()) for r in rows),max=max(len(r['caption'].split()) for r in rows),mean=round(float(np.mean([len(r['caption'].split()) for r in rows])),2)),license_counts=dict(Counter(r['license_name'] for r in rows)),category_counts=dict(category_counts.most_common()),prompt_count=len(prompts),prompt_category_count=len(covered),manifest_sha256=digest(ROOT/'data/manifest.csv'),prompts_sha256=digest(ROOT/'data/baseline_prompts.csv'),annotation_sha256={n:digest(RAW/n) for n in ['captions_val2017.json','instances_val2017.json']})
    (ROOT/'project/results/dataset_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return rows,summary
if __name__=='__main__':
    _,summary=prepare();print(json.dumps(summary,indent=2))
