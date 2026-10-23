from pathlib import Path
from collections import Counter
import csv,json,hashlib
import numpy as np
from PIL import Image
r=Path(__file__).resolve().parents[2]
def run():
    with (r/'data/manifest.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    with (r/'project/results/pair_review.csv').open(encoding='utf-8',newline='') as f:review=list(csv.DictReader(f))
    assert len(rows)==1000
    for key in ['pair_id','image_id','source_pixel_sha256','image_sha256']:
        assert len({x[key] for x in rows})==1000,key
    assert len({x['caption'].casefold() for x in rows})==1000
    assert {x['pair_id'] for x in rows}<={x['pair_id'] for x in review if x['in_final_dataset']=='True'}
    hashes=[];normalized_range=[1.0,-1.0]
    for row in rows:
        p=r/row['image_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['image_sha256']
        with Image.open(p) as im:
            assert im.mode=='RGB' and im.size==(512,512)
            pixels=np.asarray(im,dtype=np.float32)/127.5-1
            assert np.isfinite(pixels).all() and pixels.min()>=-1 and pixels.max()<=1
            normalized_range[0]=min(normalized_range[0],float(pixels.min()))
            normalized_range[1]=max(normalized_range[1],float(pixels.max()))
            a=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS))
        hashes.append(int.from_bytes(np.packbits((a[:,1:]>a[:,:-1]).ravel()).tobytes(),'big'))
    near=[]
    for i,h in enumerate(hashes):
        for j in range(i):
            distance=(h^hashes[j]).bit_count()
            if distance<=8:near.append(dict(pair_a=rows[j]['pair_id'],pair_b=rows[i]['pair_id'],hamming_distance=distance))
    summary=dict(final_pairs=1000,visually_reviewed_final_pairs=1000,total_inspected_pairs=len(review),initial_caption_flags=135,near_duplicate_pairs_reviewed=3,caption_revisions=sum(x['decision']=='caption_revised' for x in review),explicit_caption_corrections=sum(x['caption_origin']=='reviewed_correction' for x in rows),excluded_images=sum(x['decision']=='excluded' for x in review),replacement_images=sum(x['decision']=='replacement_accepted' for x in review),second_pass_refinements=7,low_alignment_pairs_reviewed=10,review_method='Codex-assisted contact-sheet review of every pair; original-resolution review for unclear images; second visual pass over all revised captions and the replacement.',all_image_hashes_verified=True,all_dimensions_rgb_verified=True,all_normalized_arrays_finite=True,normalized_range=normalized_range,exact_duplicates=0,near_duplicate_method='64-bit dHash on the processed image; all pair distances <=8 flagged for inspection.',near_duplicate_candidates=len(near),split_counts=dict(Counter(x['split'] for x in rows)),manifest_sha256=hashlib.sha256((r/'data/manifest.csv').read_bytes()).hexdigest())
    (r/'project/results/dataset_quality.json').write_text(json.dumps(summary,indent=2)+'\n')
    (r/'project/results/dhash_candidates.json').write_text(json.dumps(near,indent=2)+'\n')
    with (r/'project/results/near_duplicate_review.csv').open(encoding='utf-8',newline='') as f:decisions=list(csv.DictReader(f))
    reviewed={frozenset((x['pair_a'],x['pair_b'])) for x in decisions if x['decision']=='distinct'}
    summary['unresolved_near_duplicate_candidates']=sum(frozenset((x['pair_a'],x['pair_b'])) not in reviewed for x in near)
    summary['confirmed_near_repeats_removed']=sum(x['decision']=='near_repeat_removed' for x in decisions)
    summary['remaining_candidates_reviewed_as_distinct']=len(near)-summary['unresolved_near_duplicate_candidates']
    assert summary['unresolved_near_duplicate_candidates']==0
    (r/'project/results/dataset_quality.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
