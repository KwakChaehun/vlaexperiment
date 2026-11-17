"""Recreate pinned model/data/LIBERO assets in this checkout for an SSH worker."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess,sys,urllib.request,zipfile
from importlib import metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LIBERO_REV='8f1084e3132a39270c3a13ebe37270a43ece2a01'
MODEL_REV='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
VLM_REV='4cde9d208e19519c36917aa170fffc7e381aa0ad'
DATA_REV='a4336d589d589045d1c56423ffdf3b88a0e19b1f'


def patch_runtime():
    if sys.prefix==sys.base_prefix:
        raise RuntimeError('Runtime patches require an isolated venv; activate .venv first')
    spec=importlib.util.find_spec('lerobot')
    if spec is None:raise RuntimeError('Install lerobot==0.4.4 --no-deps first')
    if metadata.version('lerobot')!='0.4.4':raise RuntimeError('Expected pinned lerobot==0.4.4')
    target=Path(next(iter(spec.submodule_search_locations)))/'policies/__init__.py'
    original=target.read_text(encoding='utf-8')
    backup=target.with_suffix('.py.vla_original')
    if not backup.exists():backup.write_text(original,encoding='utf-8')
    # Preserve the original license header; explicit SmolVLA imports register its config.
    header=backup.read_text(encoding='utf-8').split('from .act.configuration_act')[0]
    target.write_text(header+'\n# VLA experiment: import requested policy explicitly; skip unrelated extras.\n',encoding='utf-8')
    benchmark=ROOT/'.runtime/vendor/LIBERO-master/libero/libero/benchmark/__init__.py'
    text=benchmark.read_text(encoding='utf-8')
    source='torch.load(init_states_path)'
    if source in text:
        backup=benchmark.with_suffix('.py.vla_original')
        if not backup.exists():backup.write_text(text,encoding='utf-8')
        benchmark.write_text(text.replace(source,'torch.load(init_states_path, weights_only=False)'),encoding='utf-8')
    elif 'torch.load(init_states_path, weights_only=False)' not in text:
        raise RuntimeError('Unexpected LIBERO source; refusing an unknown patch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-runtime',action='store_true',help='Apply documented compatibility patches inside an active venv')
    args=parser.parse_args()
    if sys.platform=='win32':raise RuntimeError('This SSH setup script targets Linux; existing Windows runtime is separate')
    from huggingface_hub import snapshot_download
    snapshot_download('HuggingFaceVLA/smolvla_libero',revision=MODEL_REV,token=False,
        allow_patterns=['*.json','*.safetensors','README.md'],local_dir=ROOT/'checkpoints/smolvla_libero')
    snapshot_download('HuggingFaceTB/SmolVLM2-500M-Instruct',revision=VLM_REV,token=False,
        allow_patterns=['*.json','*.txt','*.model','*.jinja'],local_dir=ROOT/'checkpoints/smolvlm_config')
    vendor=ROOT/'.runtime/vendor';vendor.mkdir(parents=True,exist_ok=True)
    archive=vendor/f'LIBERO-{LIBERO_REV}.zip';destination=vendor/'LIBERO-master'
    if not destination.exists():
        if not archive.exists():
            temporary=archive.with_suffix('.part')
            urllib.request.urlretrieve(f'https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/zip/{LIBERO_REV}',temporary)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                resolved=(vendor/member.filename).resolve()
                if not resolved.is_relative_to(vendor.resolve()):raise ValueError('Archive entry escapes vendor directory')
            z.extractall(vendor)
        extracted=vendor/f'LIBERO-{LIBERO_REV}'
        if not extracted.is_dir():raise RuntimeError('Unexpected archive layout')
        extracted.rename(destination)
    import yaml
    lib=destination/'libero/libero';configuration=ROOT/'.runtime/libero_config'
    configuration.mkdir(parents=True,exist_ok=True)
    (configuration/'config.yaml').write_text(yaml.safe_dump({
        'benchmark_root':str(lib),'bddl_files':str(lib/'bddl_files'),
        'init_states':str(lib/'init_files'),'datasets':str(ROOT/'data'),'assets':str(lib/'assets')}),encoding='utf-8')
    if args.patch_runtime:patch_runtime()
    data=ROOT/'data/libero_pilot.npz'
    if not data.exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/download_libero_subset.py'),
            '--episodes','1261','1274','1267','1269','1280','1285',
            '--max-samples-per-episode','8','--horizon','50','--revision',DATA_REV,
            '--max-download-mib','128','--output',str(data),'--cache-dir',str(ROOT/'data/hf_cache')],cwd=ROOT,check=True)
    with (ROOT/'checkpoints/smolvla_libero/model.safetensors').open('rb') as f:
        weight_hash=hashlib.file_digest(f,'sha256').hexdigest()
    manifest={'LIBERO.zip':LIBERO_REV,'model':MODEL_REV,'vlm_config':VLM_REV,'dataset':DATA_REV,
              'lerobot':'0.4.4','compatibility_patches':bool(args.patch_runtime),
              'model_sha256':weight_hash}
    (ROOT/'.runtime/assets_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
