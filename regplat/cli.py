"""命令行入口：python -m regplat run --moving m.nii.gz --fixed f.nii.gz --out out/"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from .io import load_nifti, save_nifti
from .batch import BatchRunner
from .registration import RegistrationParams


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="regplat", description="医学图像刚性配准平台")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="运行一对或一批刚性配准")
    run.add_argument("--moving", action="append", required=True, help="浮动影像 NIfTI 路径，可多次给出")
    run.add_argument("--fixed", action="append", required=True, help="固定影像 NIfTI 路径，与 --moving 一一对应")
    run.add_argument("--name", action="append", help="每对的名称（可选）")
    run.add_argument("--out", required=True, help="输出目录")
    run.add_argument("--min-ncc", type=float, default=0.6)
    run.add_argument("--min-overlap", type=float, default=0.25)
    run.add_argument("--max-iter", type=int, default=200)

    explain = sub.add_parser("explain", help="查看某批次 manifest 采用了哪份参数")
    explain.add_argument("manifest", help="manifest.json 路径")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "explain":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        print("provenance:", manifest["provenance"])
        print("params_fingerprint:", manifest["params_fingerprint"])
        print(json.dumps(manifest["params"], indent=2, ensure_ascii=False))
        return 0

    if len(args.moving) != len(args.fixed):
        raise SystemExit("--moving 与 --fixed 数量必须相同")
    names = args.name or [f"pair_{i:02d}" for i in range(len(args.moving))]
    if len(names) != len(args.moving):
        raise SystemExit("--name 数量需与 --moving 相同")
    params = RegistrationParams(
        min_ncc=args.min_ncc,
        min_overlap_fraction=args.min_overlap,
        max_iter=args.max_iter,
    )
    runner = BatchRunner(args.out, params)
    for name, mp, fp in zip(names, args.moving, args.fixed):
        moving = load_nifti(mp)
        fixed = load_nifti(fp)
        runner.run_pair(name, moving, fixed)
    manifest_path = runner.finalize("cli_batch")
    print(f"批次完成，manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
