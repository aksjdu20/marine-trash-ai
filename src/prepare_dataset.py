"""COCO/YOLO -> 통합 detection dataset. 원본은 수정하지 않습니다."""
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from src.common import (IMAGE_EXTS, NAMES, config, dataset, label_path, parser,
                        path, pixel_hash, read_image, read_labels, run, save_json,
                        split_images, write_image)


def normalize(name):
    return re.sub(r"[\s-]+", "_", str(name).strip().lower())


def csv_write(filename, rows, columns):
    with Path(filename).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def coco_records(files, image_root):
    lookup = defaultdict(list)
    for p in image_root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            lookup[p.name].append(p)
    records, categories = [], []
    for filename in files:
        d = json.loads(filename.read_text(encoding="utf-8-sig"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        if len(cats) != len(d["categories"]):
            raise ValueError(f"중복 category ID: {filename}")
        categories.extend({"annotation": str(filename), "id": k, "name": v} for k, v in cats.items())
        anns = defaultdict(list)
        ids = {i["id"] for i in d["images"]}
        if len(ids) != len(d["images"]):
            raise ValueError(f"중복 image ID: {filename}")
        for a in d["annotations"]:
            if a["image_id"] not in ids or a["category_id"] not in cats:
                raise ValueError(f"annotation 참조 ID 오류: {filename}, {a.get('id')}")
            anns[a["image_id"]].append(a)
        # 파일/부모 폴더에 명시된 공식 분할만 사용. train/val을 임의 추측하지 않음.
        tokens = re.split(r"[^a-z]+", (filename.stem + " " + filename.parent.name).lower())
        official = next((s for s in ["test", "val", "train"] if s in tokens), None)
        for im in d["images"]:
            name = im["file_name"]
            candidates = [image_root / name, filename.parent / name]
            source = next((p for p in candidates if p.is_file()), None)
            if source is None:
                hits = lookup[Path(name).name]
                if len(hits) != 1:
                    raise ValueError(f"이미지 경로 없음/모호함: {name}. --images를 해당 버전 루트로 지정하세요.")
                source = hits[0]
            labels = [(cats[a["category_id"]], a["bbox"], bool(a.get("iscrowd", 0))) for a in anns[im["id"]]]
            group = im.get("video_id", im.get("video", im.get("sequence_id")))
            records.append({"source": source.resolve(), "original_name": name, "labels": labels,
                            "group": str(group) if group is not None else None,
                            "official": im.get("split", official), "coco_size": (im["width"], im["height"]), "format": "coco"})
    return records, categories


def yolo_records(filename):
    d, _, names = dataset(filename)
    records = []
    for split in ["train", "val", "test"]:
        if not d.get(split):
            continue
        for p in split_images(filename, split):
            labels = [(names[int(row[0])], row[1:].tolist(), False) for row in read_labels(label_path(p), len(names))]
            records.append({"source": p, "original_name": p.name, "labels": labels,
                            "group": None, "official": split, "format": "yolo"})
    return records, [{"id": i, "name": n} for i, n in enumerate(names)]


def split_groups(records, seed, policy):
    """영상 그룹 보존. 공식 val만 있으면 val을 그룹 단위로 반분해 test 확보."""
    groups = defaultdict(list)
    for r in records:
        groups[r["group"]].append(r)
    if len(groups) < 3:
        raise ValueError("독립 영상/이미지 그룹이 3개 이상 필요합니다.")
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    result = {}
    if policy == "official" and all(r["official"] in {"train", "val", "test"} for r in records):
        for key, rows in groups.items():
            splits = {r["official"] for r in rows}
            if len(splits) != 1:
                raise ValueError(f"공식 분할 간 동일 영상 그룹: {key}. --split resplit로 재분할하세요.")
            result[key] = next(iter(splits))
        if "test" not in result.values():
            vals = [k for k in keys if result[k] == "val"]
            if len(vals) < 2:
                raise ValueError("공식 val 영상 그룹이 2개 미만. --split resplit 필요")
            # ponytail: 그룹 수를 반분. 영상 길이가 다르면 이미지 비율은 정확히 15%가 아님.
            for k in vals[:max(1, len(vals)//2)]:
                result[k] = "test"
    else:
        # ponytail: 그룹 수 기준 70/15/15 근사. 크기가 큰 영상은 비율을 바꿀 수 있음.
        nt = max(1, round(len(keys)*0.15))
        nv = max(1, round(len(keys)*0.15))
        for i, key in enumerate(keys):
            result[key] = "test" if i < nt else "val" if i < nt+nv else "train"
    if set(result.values()) != {"train", "val", "test"}:
        raise ValueError("train/val/test 중 빈 분할이 발생했습니다. 영상 그룹/분할을 확인하세요.")
    for r in records:
        r["split"] = result[r["group"]]


def prepare(args):
    out = path(args.output)
    if out.exists() and any(out.iterdir()) and list(out.iterdir()) != [out / ".gitkeep"]:
        raise ValueError(f"출력 폴더가 비어 있지 않습니다: {out}. 새로운 --output 경로를 사용하세요.")
    out.mkdir(parents=True, exist_ok=True)
    if args.yolo:
        records, categories = yolo_records(args.yolo)
    else:
        if not args.annotations or not args.images:
            raise ValueError("COCO는 --annotations JSON [JSON ...] --images 폴더가 필요합니다.")
        records, categories = coco_records([path(p) for p in args.annotations], path(args.images))
    csv_write(out / "categories.csv", categories, sorted(set().union(*(r.keys() for r in categories))))
    print("실제 원본 클래스:", sorted({c["name"] for c in categories}))
    if args.inspect:
        print(f"클래스 목록 저장: {out/'categories.csv'}. 매핑을 검토한 뒤 새로운 출력 폴더로 변환하세요.")
        return out
    mapping = {normalize(k): v for k, v in yaml.safe_load(path(args.mapping).read_text(encoding="utf-8")).items()}
    names = NAMES[:2] if args.two_classes else NAMES
    group_map = {}
    if args.groups:
        with path(args.groups).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("video_id") or not row.get("file_name"):
                    raise ValueError("groups.csv는 file_name,video_id 열과 비어 있지 않은 값이 필요합니다.")
                if row["file_name"] in group_map and group_map[row["file_name"]] != row["video_id"]:
                    raise ValueError("groups.csv의 중복 file_name에 서로 다른 영상 ID")
                group_map[row["file_name"]] = row["video_id"]
    review, clean, duplicates, adjustments, label_duplicates = [], [], [], [], []
    by_hash = {}
    aliases = {}
    # 동일 픽셀이 서로 다른 영상 ID로 기록되면 그룹을 합쳐 누출을 방지.
    def root(g):
        while g in aliases:
            g = aliases[g]
        return g
    for r in records:
        try:
            im = read_image(r["source"])
            h, w = im.shape[:2]
            if r.get("coco_size", (w,h)) != (w,h):
                raise ValueError("COCO width/height와 실제 이미지 크기가 다릅니다")
            labels = []
            for name, box, crowd in r["labels"]:
                target = mapping.get(normalize(name))
                if target not in names:
                    raise ValueError(f"검토/제외 클래스: {name} -> {target}. 객체를 무시한 채 배경으로 학습하지 않도록 이미지 전체 제외")
                if crowd:
                    raise ValueError("iscrowd annotation 검토 필요 (YOLO detection에 ignore region 없음)")
                if r["format"] == "coco":
                    x, y, bw, bh = [float(v) for v in box]
                    if getattr(args, "clip_border", False):
                        overflow = max(0, -x, -y, x+bw-w, y+bh-h)
                        if overflow > 1+1e-6:
                            raise ValueError("bbox 경계 초과가 1픽셀보다 큼: 자동 수정하지 않고 검토")
                        if overflow > 0:
                            original = [x,y,bw,bh]
                            x2,y2 = min(w,x+bw),min(h,y+bh)
                            x,y = max(0,x),max(0,y)
                            bw,bh = x2-x,y2-y
                            adjustments.append({"source":str(r["source"]),"class":name,"original_bbox":str(original),
                                                "clipped_bbox":str([x,y,bw,bh]),"overflow_pixels":overflow})
                    box = [(x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h]
                labels.append([names.index(target), *box])
            # 동일 validator를 사용하기 전에 임시 파일 없이 경계 확인
            for c, x, y, bw, bh in labels:
                import math
                if not all(math.isfinite(v) for v in [c,x,y,bw,bh]) or bw <= 0 or bh <= 0 or min(x-bw/2,y-bh/2) < -1e-6 or max(x+bw/2,y+bh/2) > 1+1e-6:
                    raise ValueError("잘못된 bbox 좌표")
            # 같은 detection 클래스/좌표는 원본 polygon이 달라도 한 정답 박스입니다.
            canonical = [tuple(round(float(v),8) for v in row) for row in labels]
            unique = sorted(set(canonical))
            if len(unique) != len(labels):
                label_duplicates.append({"source":str(r["source"]),"removed":len(labels)-len(unique),
                                         "reason":"identical mapped class and bbox at YOLO output precision"})
            r["converted"] = unique
            r["hash"] = pixel_hash(im)
            r["width"], r["height"] = w, h
            r["group"] = group_map.get(r["original_name"], r["group"])
            if r["group"] is None and args.group_regex:
                m = re.search(args.group_regex, r["original_name"])
                if m is None or m.lastindex is None:
                    raise ValueError("--group-regex가 일치하지 않음. 영상 ID를 캡처하는 괄호가 필요")
                r["group"] = m.group(1)
            if r["group"] is None:
                if not args.allow_image_split:
                    raise ValueError("영상 ID 없음. --groups/--group-regex를 제공하거나 독립 사진만 --allow-image-split 사용")
                r["group"] = "image:" + r["hash"]
            r["group"] = str(r["group"])
            if r["hash"] in by_hash:
                prior = by_hash[r["hash"]]
                if prior["converted"] != r["converted"]:
                    raise ValueError("동일 이미지의 라벨이 상충함. Material/Instance 중 하나만 사용하거나 라벨을 직접 검토")
                duplicates.append({"source": str(r["source"]), "kept": str(prior["source"]), "hash": r["hash"]})
                a, b = root(r["group"]), root(prior["group"])
                if a != b:
                    aliases[a] = b
                # 공식 split 충돌은 기록/차단 (resplit 선택 시 그룹으로 다시 분할)
                if args.split == "official" and prior["official"] != r["official"]:
                    raise ValueError("동일 이미지가 공식 분할 여러 곳에 존재. --split resplit 필요")
                continue
            by_hash[r["hash"]] = r
            clean.append(r)
        except (ValueError, OSError, KeyError, TypeError) as e:
            review.append({"source": str(r["source"]), "reason": str(e)})
    csv_write(out/"review.csv", review, ["source","reason"])
    csv_write(out/"duplicates.csv", duplicates, ["source","kept","hash"])
    csv_write(out/"bbox_adjustments.csv", adjustments, ["source","class","original_bbox","clipped_bbox","overflow_pixels"])
    csv_write(out/"label_duplicates.csv", label_duplicates, ["source","removed","reason"])
    # 기본은 문제 해결 후 다시 실행. --skip-review는 제외 이미지 수를 실험 기록에 남김.
    if review and not args.skip_review:
        raise ValueError(f"검토 대상 {len(review)}개. {out/'review.csv'} 확인 후 수정하거나 --skip-review + 새로운 --output 사용")
    for r in clean:
        r["group"] = root(r["group"])
    split_groups(clean, args.seed, args.split)
    manifest = []
    for r in clean:
        filename = r["hash"] + ".png"
        target = out / "images" / r["split"] / filename
        write_image(target, read_image(r["source"]))
        lab = out / "labels" / r["split"] / Path(filename).with_suffix(".txt")
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text("".join(" ".join(f"{v:.8f}" if i else str(int(v)) for i,v in enumerate(row))+"\n" for row in r["converted"]), encoding="utf-8")
        manifest.append({"image": str(target.relative_to(out)).replace("\\","/"), "source": str(r["source"]),
                         "video_id": r["group"], "split": r["split"], "official_split": r["official"],
                         "pixel_sha256": r["hash"], "width": r["width"], "height": r["height"]})
    csv_write(out/"manifest.csv", manifest, list(manifest[0]))
    (out/"dataset.yaml").write_text(yaml.safe_dump({"path": str(out), "train":"images/train", "val":"images/val", "test":"images/test", "names": names}, sort_keys=False), encoding="utf-8")
    save_json(out/"preparation.json", {"seed":args.seed, "split_policy":args.split, "groups_file":args.groups,
                                      "group_regex":args.group_regex,"image_split_allowed":args.allow_image_split,
                                      "mapping":mapping,"review_excluded":len(review),"duplicates_removed":len(duplicates),
                                      "clip_border":getattr(args,"clip_border",False),"bbox_adjustments":len(adjustments),
                                      "duplicate_labels_removed":sum(row["removed"] for row in label_duplicates),
                                      "source_annotations":args.annotations,"source_yolo":args.yolo,
                                      "note":"영상 그룹 수 기준 분할. 실제 이미지 비율은 manifest/counts 참조. 누락 객체는 육안 검토 필요."})
    from src.validate_dataset import validate
    validate(out/"dataset.yaml", out/"analysis")
    print(f"준비 완료: {out/'dataset.yaml'}")
    return out


def main():
    p = parser(__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", nargs="+")
    source.add_argument("--yolo", help="YOLO data.yaml. names는 원본 클래스")
    p.add_argument("--images")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--mapping", default="class_mapping.yaml")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--groups", help="file_name,video_id CSV")
    p.add_argument("--group-regex")
    p.add_argument("--allow-image-split", action="store_true")
    p.add_argument("--skip-review", action="store_true")
    p.add_argument("--clip-border", action="store_true",help="COCO bbox의 1픽셀 이내 경계 초과만 잘라내고 CSV에 기록")
    p.add_argument("--two-classes", action="store_true")
    p.add_argument("--split", choices=["official","resplit"], default="official")
    p.add_argument("--seed", type=int, default=42)
    prepare(p.parse_args())


if __name__ == "__main__":
    run(main)
