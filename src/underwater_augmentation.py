"""OpenCV/NumPy 호환 증강: 좌표를 바꾸지 않는 광학 효과만 적용."""
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.common import (config, dataset, hash_file, label_path, parser, path,
                        read_image, run, save_json, split_images, write_image)


def augment(image, rng, strength=0.5, probability=0.8):
    if not 0 <= strength <= 1 or not 0 <= probability <= 1:
        raise ValueError("증강 strength/probability는 0~1")
    if rng.random() >= probability or strength == 0:
        return image.copy()
    out = image.astype(np.float32)
    contrast = rng.uniform(1-0.4*strength,1+0.4*strength)
    brightness = rng.uniform(-80,50)*strength
    out = (out-127.5)*contrast+127.5+brightness
    tint = rng.uniform(0,0.35)*strength
    if rng.random() < 0.5:
        out *= np.array([1+tint,1+tint/3,1-tint])  # BGR blue
    else:
        out *= np.array([1+tint/3,1+tint,1-tint])
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out,(3,3),0.3+1.2*strength)
    out += rng.normal(0,5*strength,out.shape)
    return np.clip(out,0,255).astype(np.uint8)


def build_training_copy(data, output, seed, settings):
    d, base, names = dataset(data)
    output = Path(output)
    if output.exists():
        raise ValueError(f"증강 출력 이미 존재: {output}. 다른 run 이름을 사용하세요.")
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    images = split_images(data,"train")
    for i, p in enumerate(images):
        dest = output/"images"/"train"/f"{i:06d}.png"
        im = read_image(p)
        altered = augment(im,rng,**settings)
        write_image(dest,altered)
        lab = output/"labels"/"train"/f"{i:06d}.txt"
        lab.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(label_path(p),lab)
        if i < 8:
            write_image(output/"previews"/f"{i:06d}.jpg",np.hstack([im,altered]))
    new = {"path":str(output.resolve()),"train":"images/train","names":names}
    # val/test는 원본 파일 목록. 절대 복제/증강하지 않음.
    for split in ["val","test"]:
        if d.get(split):
            listing = output/f"{split}.txt"
            listing.write_text("\n".join(str(p) for p in split_images(data,split))+"\n",encoding="utf-8")
            new[split] = str(listing.resolve())
    filename = output/"dataset.yaml"
    filename.write_text(yaml.safe_dump(new,sort_keys=False),encoding="utf-8")
    save_json(output/"augmentation.json",{"seed":seed,"settings":settings,"source":str(path(data)),"source_count":len(images),
                                           "method":"offline 1:1 photometric replacement, fixed per seed; same object boxes and number of images",
                                           "originals":[{"path":str(p),"sha256":hash_file(p)} for p in images]})
    return filename


def main():
    p = parser(__doc__)
    p.add_argument("--image",required=True)
    p.add_argument("--output",default="results/augmentation_preview.jpg")
    args = p.parse_args()
    cfg = config(args.config)
    im = read_image(path(args.image))
    altered = augment(im,np.random.default_rng(cfg["seed"]),**cfg["underwater"])
    write_image(path(args.output),np.hstack([im,altered]))
    print(path(args.output))


if __name__ == "__main__":
    run(main)
