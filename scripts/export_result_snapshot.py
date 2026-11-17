"""Export small versioned reports/raw metrics; omit models, input data, GIFs and logs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--dest',required=True);args=p.parse_args()
    source=Path(args.source).resolve();dest=Path(args.dest).resolve()
    if source==dest or source in dest.parents:raise ValueError('Snapshot must be outside the live source directory')
    dest.mkdir(parents=True,exist_ok=True)
    hashes={}
    excluded={'launch.json','current_episode.json','latest_view.png','metadata_corrections.json','offline.before_N_metadata_fix.jsonl'}
    for path in sorted(source.rglob('*')):
        relative=path.relative_to(source)
        if not path.is_file() or path.name in excluded or any(x in {'replays','arrival_replays','.cache'} for x in relative.parts):continue
        if path.suffix not in {'.json','.jsonl','.md','.html','.png','.txt'} or path.name.endswith('.log'):continue
        if path.stat().st_size>10*1024**2:continue
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if target.suffix=='.html':
            text=target.read_text(encoding='utf-8').replace('<meta http-equiv="refresh" content="60">','')
            target.write_text('<div style="padding:16px;background:#fff3cd">Frozen exported snapshot; see snapshot.json for export time. Live results may have advanced.</div>'+text,encoding='utf-8')
        hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    exported=datetime.now(timezone.utc).isoformat()
    (dest/'snapshot.json').write_text(json.dumps({'exported_utc':exported,'source':source.name,'frozen_snapshot':True,
        'omitted':'model/input dataset/venv/runtime/GIFs/logs','files_sha256':hashes},indent=2),encoding='utf-8')
    (dest/'README.md').write_text(f'# Frozen result snapshot\n\nExported UTC: {exported}.\n\nThis is an exported snapshot, not a live worker. A `running` status describes the source at export time.\n\n[Report](RESULTS.md) / [HTML](RESULTS.html).\n\nModel weights, input datasets, runtime environments, GIFs and logs are omitted. Public model/data revisions are in manifest.json; recreate them using docs/SSH_GPU_RUNBOOK.md.\n',encoding='utf-8')
    print(json.dumps({'source':source.name,'files':len(hashes),'destination':str(dest)},indent=2))


if __name__=='__main__':main()
