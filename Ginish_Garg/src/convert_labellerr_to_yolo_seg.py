"""
If Labellerr export is COCO, Ultralytics can train directly via data/coco.yaml.
Use this script ONLY if you must convert to YOLO TXT with polygons.
"""
import json, os, shutil
from pathlib import Path

def main(coco_json, out_base="data/yolo_data"):
    print("Tip: prefer training with COCO directly via Ultralytics and skip conversion.")
    # parse JSON and write YOLO txts if absolutely required (left as TODO).

if __name__ == "__main__":
    import sys
    if len(sys.argv)<2:
        print("Usage: python src/convert_labellerr_to_yolo_seg.py path/to/annotations.json")
        raise SystemExit(1)
    main(sys.argv[1])
