"""labelme JSON → YOLO-seg txt（原地转换，不复制图像）。

流程中的位置：切片完成、``images/`` 与 ``labels/`` 分好目录之后，
把 ``labels/*.json`` 转成同名的 ``labels/*.txt``（两者并存，json 保留作溯源）。

输出格式（YOLO instance-seg）：``class_id x1 y1 x2 y2 ...``，坐标按图像宽高归一化、
保留 6 位小数、逐点 clamp 到 [0,1]。无标注的图会写出**空 txt**（这是有意的：
训练集需要负样本，空文件即"该瓦片无目标"）。

类别：默认 ``0=landslide, 1=road``。``damage`` 在本项目最终口径里已移除
（原始 val 中仅约 16.2 万像素、占比 ~0.005%，模型学不到且会拖低 mIoU 均值），
所以默认**跳过** damage 标注并计数；确需复现旧数据加 ``--keep-damage``。

用法::

    python data_pipeline/preprocess/labelme2yolo.py --root ./data/DPT_region
    python data_pipeline/preprocess/labelme2yolo.py --root ./data/DPT_region --region Wenma
"""
import os
import json
import argparse

# 最终口径：只保留两类；damage 默认丢弃（见模块 docstring）
CLASS_MAP = {"landslide": 0, "road": 1}
CLASS_MAP_WITH_DAMAGE = {"landslide": 0, "road": 1, "damage": 2}


def convert_labels_dir(labels_dir):
    n_json = 0
    n_txt = 0
    n_inst = 0
    n_skip_label = 0
    n_empty = 0
    for jf in sorted(os.listdir(labels_dir)):
        if not jf.lower().endswith(".json"):
            continue
        n_json += 1
        jp = os.path.join(labels_dir, jf)
        with open(jp, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        w = data.get("imageWidth")
        h = data.get("imageHeight")
        lines = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue
            cid = CLASS_MAP.get(shape.get("label"))
            if cid is None:
                n_skip_label += 1
                continue
            pts = []
            for x, y in shape["points"]:
                xn = min(max(x / w, 0.0), 1.0)
                yn = min(max(y / h, 0.0), 1.0)
                pts.append(f"{xn:.6f}")
                pts.append(f"{yn:.6f}")
            lines.append(f"{cid} " + " ".join(pts))
            n_inst += 1
        txt_path = os.path.join(labels_dir, os.path.splitext(jf)[0] + ".txt")
        with open(txt_path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
            if lines:
                fp.write("\n")
        n_txt += 1
        if not lines:
            n_empty += 1
    return n_json, n_txt, n_inst, n_skip_label, n_empty


def main():
    global CLASS_MAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="瓦片数据集根目录：{root}/{region}/{split}/labels/*.json")
    ap.add_argument("--region", default=None, help="仅转换指定区域")
    ap.add_argument("--keep-damage", action="store_true",
                    help="保留 damage 类（class_id=2）；默认跳过，与论文口径一致")
    args = ap.parse_args()
    if args.keep_damage:
        CLASS_MAP = CLASS_MAP_WITH_DAMAGE

    regions = sorted(
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d)) and not d.startswith("_")
    )
    if args.region:
        regions = [r for r in regions if r == args.region]
        if not regions:
            print(f"[ERROR] region not found: {args.region}")
            return

    tot = [0, 0, 0, 0, 0]
    for region in regions:
        for split in ("train", "val"):
            ldir = os.path.join(args.root, region, split, "labels")
            if not os.path.isdir(ldir):
                print(f"[MISS] {region}/{split}")
                continue
            r = convert_labels_dir(ldir)
            print(f"{region}/{split}: json={r[0]} txt={r[1]} inst={r[2]} "
                  f"skip_label={r[3]} empty_txt={r[4]}")
            tot = [a + b for a, b in zip(tot, r)]
    print("-" * 60)
    print(f"TOTAL: json={tot[0]} txt={tot[1]} inst={tot[2]} "
          f"skip_label={tot[3]} empty_txt={tot[4]}")
    # 写统一 classes.txt
    with open(os.path.join(args.root, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(CLASS_MAP, key=CLASS_MAP.get)) + "\n")
    print(f"classes.txt -> {os.path.join(args.root, 'classes.txt')}")
    print("DONE.")


if __name__ == "__main__":
    main()
