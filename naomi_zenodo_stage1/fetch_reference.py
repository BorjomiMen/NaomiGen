"""Fetch verified ranges of the uncompressed BigTIFF, without the 3.86 GB download."""
from pathlib import Path
import argparse, hashlib, json, struct, time, urllib.request
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
URL = 'https://zenodo.org/records/6293696/files/01_ZebrafishMul_GCaMP6s_485x400x9800_highSNR.tif?download=1'
STRIDE, FIRST_IFD, SIZE = 393806, 392866, 3859298816

def get_range(start, stop):
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{stop-1}'})
            with urllib.request.urlopen(req, timeout=90) as r:
                assert r.status == 206, f'Range request not honored: {r.status}'
                assert r.headers['Content-Range'] == f'bytes {start}-{stop-1}/{SIZE}'
                out = r.read()
            assert len(out) == stop-start
            return out
        except Exception:
            if attempt == 3: raise
            time.sleep(2**attempt)

def decode_page(block, page):
    start = 16 + page*STRIDE
    local_ifd = FIRST_IFD + page*STRIDE - start
    n, = struct.unpack_from('<Q', block, local_ifd)
    tags = {t[0]: t[1:] for t in (struct.unpack_from('<HHQQ',block,local_ifd+8+20*i) for i in range(n))}
    assert tags[256][2] == 485 and tags[257][2] == 400
    assert tags[258][2] == 16 and tags[259][2] == 1 and tags[339][2] == 2
    count = tags[273][1]
    offsets = np.frombuffer(block, '<u8', count=count, offset=tags[273][2]-start)
    lengths = np.frombuffer(block, '<u8', count=count, offset=tags[279][2]-start)
    assert np.array_equal(offsets, start + np.arange(count)*8730)
    assert np.all(lengths == 8730)
    nxt, = struct.unpack_from('<Q',block,local_ifd+8+20*n)
    assert nxt == (FIRST_IFD+(page+1)*STRIDE if page<9799 else 0)
    return np.frombuffer(block, '<i2', count=400*485).reshape(400,485).copy()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--refresh',action='store_true'); args=parser.parse_args()
    target=ROOT/'reference'; target.mkdir(exist_ok=True)
    out=target/'highSNR_sample.tif'
    if out.exists() and not args.refresh:
        print(f'Cached: {out}'); return
    header=get_range(0,16)
    assert header == b'II+\x00\x08\x00\x00\x00' + struct.pack('<Q',FIRST_IFD)
    frames=[]; indices=[]; windows=[]
    starts=np.linspace(0,9800-16,8).astype(int)
    for window,start_page in enumerate(starts):
        start_page=int(start_page)
        data=get_range(16+start_page*STRIDE,16+(start_page+16)*STRIDE)
        for k in range(16):
            p=start_page+k
            frames.append(decode_page(data[k*STRIDE:(k+1)*STRIDE],p))
            indices.append(p); windows.append(window)
        print(f'Window {window+1}/8, source frames {start_page}..{start_page+15}',flush=True)
    arr=np.stack(frames)
    tifffile.imwrite(out,arr,photometric='minisblack',metadata={'axes':'TYX'})
    provenance={'source_url':URL,'record_doi':'10.5281/zenodo.6293696',
        'source_total_bytes':SIZE,'source_advertised_md5':'06bb336e85d4890ed86528015c1ce958',
        'full_source_md5_verified':False,'access':'HTTP byte ranges; every decoded page IFD and strip layout verified',
        'sample_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'shape':list(arr.shape),'dtype':str(arr.dtype),'source_frame_indices_zero_based':indices,
        'window_ids':windows,'pixel_size_um':0.873,'fps':15,
        'sampling_note':'8 evenly spaced 16-frame windows from one recording; not independent animals',
        'optical_source':'https://cabooster.github.io/DeepCAD-RT/Datasets/'}
    (target/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'shape':list(arr.shape),'min':int(arr.min()),'max':int(arr.max()),'mean':float(arr.mean())}))

if __name__=='__main__': main()
