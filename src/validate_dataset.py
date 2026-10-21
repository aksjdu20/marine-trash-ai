"""이미지/라벨/중복/영상 누출 점검 및 실제 분포 시각화."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import (COLORS, NAMES, dataset, draw, hash_file, label_path, parser,
                        path, pixel_hash, read_image, read_labels, run, save_json,
                        split_images, write_image, xyxy)


def validate(data, output, require_test=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    d, base, names = dataset(data)
    if names not in [NAMES[:2], NAMES]:
        raise ValueError(f"클래스 이름/순서 오류: {names}")
    errors, rows, counts, seen, records = [], [], Counter(), {}, []
    manifest = {}
    if (base/"manifest.csv").is_file():
        with (base/"manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            manifest = {str((base/r["image"]).resolve()):r for r in csv.DictReader(f)}
    group_splits = defaultdict(set)
    digests = []
    splits = ["train","val","test"] if require_test else ["train","val"]
    for split in splits:
        try:
            images = split_images(data, split)
        except (ValueError, OSError) as e:
            errors.append({"image":split,"reason":str(e)})
            continue
        for p in images:
            try:
                im = read_image(p)
                labels = read_labels(label_path(p), len(names))
                h, w = im.shape[:2]
                digest = pixel_hash(im)
                if digest in seen:
                    raise ValueError(f"중복 픽셀 이미지: {seen[digest]} <-> {split}/{p.name}")
                seen[digest] = f"{split}/{p.name}"
                m = manifest.get(str(p.resolve()))
                if manifest and m is None:
                    raise ValueError("manifest.csv에 이미지 누락")
                if m:
                    group_splits[m["video_id"]].add(split)
                    if m["split"] != split or m["pixel_sha256"] != digest:
                        raise ValueError("manifest 분할/해시와 실제 데이터가 다름")
                if len(labels) != len({tuple(row) for row in labels}):
                    raise ValueError("완전히 중복된 bbox 라벨")
                cnt = Counter(int(v) for v in labels[:,0])
                for c, name in enumerate(names):
                    counts[(split,name,"images")] += int(cnt[c] > 0)
                    counts[(split,name,"objects")] += cnt[c]
                rows.append({"split":split,"image":str(p),"width":w,"height":h,"objects":len(labels),"background":int(not len(labels)),"pixel_sha256":digest})
                digests.append((split, digest, hash_file(label_path(p)), m["video_id"] if m else "unknown"))
                records.append((split, p, im, labels)) if sum(x[0] == split for x in records) < 4 else None
            except (ValueError, OSError) as e:
                errors.append({"image":str(p),"reason":str(e)})
    for g, s in group_splits.items():
        if len(s) > 1:
            errors.append({"image":g,"reason":f"영상 그룹 분할 누출: {sorted(s)}"})
    pd.DataFrame(errors, columns=["image","reason"]).to_csv(output/"errors.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output/"images.csv",index=False,encoding="utf-8-sig")
    dist = pd.DataFrame([{"split":s,"class":n,"images":counts[(s,n,"images")],"objects":counts[(s,n,"objects")]} for s in splits for n in names])
    dist.to_csv(output/"class_counts.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"split":s,"images":sum(r["split"]==s for r in rows)} for s in splits]).to_csv(output/"split_counts.csv",index=False,encoding="utf-8-sig")
    ax = dist.pivot(index="class",columns="split",values="objects").plot.bar(rot=0, title="Object counts by class / split")
    ax.set_ylabel("Labeled objects")
    ax.figure.tight_layout()
    ax.figure.savefig(output/"class_distribution.png",dpi=160)
    plt.close(ax.figure)
    for split, p, im, lab in records:
        import numpy as np
        boxes = np.column_stack([xyxy(lab, im.shape[1], im.shape[0]), np.ones(len(lab)), lab[:,0]])
        write_image(output/"samples"/f"{split}_{p.stem}.jpg",draw(im,boxes,names))
    import hashlib, json
    fingerprint = hashlib.sha256(json.dumps(sorted(digests),ensure_ascii=False).encode()).hexdigest()
    report = {"valid":not errors,"errors":len(errors),"images":len(rows),"names":names,"fingerprint":fingerprint,
              "video_groups_verified":bool(manifest),"note":"라벨 누락 객체/의미적 오라벨과 JPEG 재압축/근접 프레임 중복은 자동 판별 불가. 샘플 육안 검토 필수."}
    save_json(output/"validation.json",report)
    if errors:
        raise ValueError(f"데이터 검증 실패 {len(errors)}건: {output/'errors.csv'}")
    print(f"검증 통과: {len(rows)} 이미지, 영상 그룹 확인={bool(manifest)}")
    return report


def main():
    p = parser(__doc__)
    p.add_argument("--data")
    p.add_argument("--output",default="results/dataset")
    args = p.parse_args()
    from src.common import config
    validate(path(args.data or config(args.config)["data"]),path(args.output))


if __name__ == "__main__":
    run(main)
