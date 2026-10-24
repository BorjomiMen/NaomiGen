"""NAOMi -> calibrated still images + aligned instance/semantic masks."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, sys, time
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.io import loadmat
from scipy.optimize import least_squares
from skimage.segmentation import find_boundaries
from PIL import Image,ImageDraw,ImageFont
from measure_reference import detect,image_metrics,ROOT,N,PIX
from runtime_paths import octave_runtime,plot_font

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write_json(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def ensure_anatomy(seed,tag='v5'):
    path=ROOT/'anatomy'/f'{tag}_seed{seed}'/'footprints.mat'
    if not path.exists() and not (path.parent/'scanner_probes.mat').exists():
        executable,env=octave_runtime()
        octave_path=ROOT.as_posix().replace("'","''")
        command=f"addpath('{octave_path}');generate_stage1_anatomy({seed},struct('tag','{tag}'));"
        subprocess.run([executable,'--quiet','--no-gui','--eval',command],env=env,check=True)
    if not path.exists():
        from sparse_scanner import build_footprints
        build_footprints(path.parent)
    return path

def load_anatomy(seed,tag='v5'):
    p=ensure_anatomy(seed,tag)
    a=loadmat(p,simplify_cells=True)
    fp=np.maximum(a['footprints'].astype(np.float32),0)
    assert fp.ndim==3 and np.isfinite(fp).all()
    return fp,np.asarray(a['locs']),p

def sample_basis(fp,cfg,dx=0,dy=0):
    # Native NAOMi scanner pixels are 1 um, with its original 5 um edge crop.
    # Coordinate mapping preserves 0.873 um sampling, rather than resizing by eye.
    yy=(np.arange(N)-(N-1)/2)*PIX+(fp.shape[0]-1)/2+dy
    xx=(np.arange(N)-(N-1)/2)*PIX+(fp.shape[1]-1)/2+dx
    yy,xx=np.meshgrid(yy,xx,indexing='ij')
    out=np.stack([ndi.map_coordinates(fp[:,:,k],[yy,xx],order=1,mode='nearest') for k in range(fp.shape[2])],axis=-1)
    if cfg['additional_blur_px']>0:
        out=ndi.gaussian_filter(out,(cfg['additional_blur_px'],cfg['additional_blur_px'],0))
    return out/cfg['footprint_reference_peak']

def latent(fp,cfg,rng,dx=0,dy=0):
    basis=sample_basis(fp,cfg,dx,dy)
    k=basis.shape[-1]
    weights=rng.lognormal(-cfg['expression_log_sigma']**2/2,cfg['expression_log_sigma'],k)
    weights*=rng.random(k)<cfg['labelled_fraction']
    components=basis*weights
    signal=components.sum(-1)
    return signal,components

def camera(clean,cfg,rng,repeats=1):
    offset=cfg['camera_offset_adu']; gain=cfg['shot_gain_adu']
    signal=np.maximum(clean-offset,0)
    noisy=offset+gain*rng.poisson(signal/gain,size=(repeats,*signal.shape))
    noisy+=rng.normal(0,cfg['effective_gaussian_sigma_adu'],noisy.shape)
    return np.rint(np.clip(noisy,0,cfg['saturation_adu'])).astype(np.uint16)

def moments(candidates):
    out={}
    for key in candidates[0]:out[key]=float(np.median([r[key] for r in candidates]))
    return out

def calibrate(tag='v5'):
    ref=read_json(ROOT/'reference'/'measurements.json')
    rows=[r for r in ref['rows'] if r['window']<5]
    target=moments([r['single_frame'] for r in rows])
    mean_target=moments([r['mean_image'] for r in rows])
    fp,locs,_=load_anatomy(21,tag)
    middle=np.abs(locs[:,2]-25)<2
    normalizer=float(np.median(fp[:,:,middle].max(axis=(0,1))))
    base={'version':1,'anatomy_tag':tag,'crop_size_px':N,'pixel_size_um':PIX,
        'volume_um':[80,80,50],'cell_slab_um':20,'neurons_3d':600,'cell_radius_um':2.2,'nucleus_radius_um':1.5,
        'min_distance_um':5.0,'nuclear_label':0,'nuclear_fill_fraction':.1,'native_pixel_um':1.,
        'additional_blur_px':0.,'labelled_fraction':1.,'expression_log_sigma':.45,
        'footprint_reference_peak':normalizer,'camera_offset_adu':17800.,
        'shot_gain_adu':ref['noise_model']['variance_slope_adu'],
        'effective_gaussian_sigma_adu':np.sqrt(ref['noise_model']['variance_intercept_adu2']),
        'saturation_adu':32767,'gain_adu':1.,'background_adu':0.,
        'mask_relative_threshold':.5,'mask_min_peak_adu':2000.,'mask_min_area_px':5,'mask_max_area_px':60,
        'fit_source_rois':['A','B','C','D'],'fit_source_windows':[0,1,2,3,4],'fit_anatomy_seed':21,
        'scope':'Dense local brain patches from one Zenodo recording. Approximate image morphology, not recovered 3D biology.',
        'independence':'Images from the same anatomy seed share geometry; split datasets by anatomy seed.'}
    target_count=ref['fit_proxy_count_median']; target_d=ref['fit_proxy_diameter_um_q10_q50_q90'][1];target_nn=ref['fit_proxy_nn_um_q10_q50_q90'][1]
    records=[]
    for fraction,blur,expression_sigma in itertools.product([.85,1.],[0.,.35],[.15,.3,.45]):
        c={**base,'additional_blur_px':blur,'labelled_fraction':fraction,'expression_log_sigma':expression_sigma}
        signals=[]
        for j in range(4):
            signal,_=latent(fp,c,np.random.default_rng(210000+j),dx=(j%2)*3-1.5,dy=(j//2)*3-1.5)
            signals.append(signal)
        raw=np.stack(signals)
        variance_noise=c['shot_gain_adu']*(mean_target['mean']-c['camera_offset_adu'])+c['effective_gaussian_sigma_adu']**2
        struct_std=np.sqrt(max(1,mean_target['std']**2-variance_noise/16))
        gain=struct_std/np.median([s.std() for s in signals])
        bg=max(0,mean_target['mean']-c['camera_offset_adu']-gain*raw.mean())
        # The camera parameters are a single fixed affine calibration. No per-image histogram matching.
        def residual(x):
            c2={**c,'gain_adu':float(x[0]),'background_adu':float(x[1])}
            predictions=[]
            for j,s in enumerate(signals):
                q=camera(c['camera_offset_adu']+x[1]+x[0]*s,c2,np.random.default_rng(9000+j),repeats=16)
                predictions.append(image_metrics(q[0]))
            m=moments(predictions)
            return np.array([(m['mean']-target['mean'])/1000,(m['std']-target['std'])/1000,
                (m['p10']-target['p10'])/2000,(m['p90']-target['p90'])/2000])
        # Small deterministic grid is preferable to differentiating stochastic Poisson counts.
        choices=[]
        for gs in [.85,1.,1.15]:
          for bs in [-400,0,400]:
            xx=[gain*gs,max(0,bg+bs)]; rr=residual(xx);choices.append((float(rr@rr),xx))
        _,best=min(choices,key=lambda x:x[0]);c['gain_adu'],c['background_adu']=map(float,best)
        measured=[]
        for j,s in enumerate(signals):
            q=camera(c['camera_offset_adu']+c['background_adu']+c['gain_adu']*s,c,np.random.default_rng(9000+j),16)
            _,d=detect(q.mean(0)); measured.append(d)
        count=float(np.median([m['count'] for m in measured]));diam=float(np.median([m['diameter_median_um'] for m in measured]));nn=float(np.median([m['nn_median_um'] for m in measured]))
        score=abs(np.log(count/target_count))+abs(np.log(diam/target_d))+abs(np.log(nn/target_nn))+.15*min(choices,key=lambda x:x[0])[0]
        rec={'score':float(score),'fraction':fraction,'blur':blur,'expression_sigma':expression_sigma,'count':count,'diameter_um':diam,'nn_um':nn,'cfg':c}
        records.append(rec);print('Candidate',json.dumps({k:v for k,v in rec.items() if k!='cfg'}),flush=True)
    winner=min(records,key=lambda x:x['score']);cfg=winner['cfg']
    cfg['calibration_target']=target;cfg['calibration_target_mean16']=mean_target
    provenance=read_json(ROOT/'reference'/'provenance.json')
    window_frames=[provenance['source_frame_indices_zero_based'][i*16] for i in range(8)]
    cfg['brightness_reference_frame']=window_frames[2]
    fit_frames=np.array([window_frames[r['window']] for r in rows],float)
    fit_log_signal=np.log([r['mean_image']['mean']-cfg['camera_offset_adu'] for r in rows])
    slope,intercept=np.polyfit(fit_frames-cfg['brightness_reference_frame'],fit_log_signal,1)
    cfg['log_brightness_slope_per_source_frame']=float(slope)
    noise_variances=np.array([cfg['shot_gain_adu']*(r['mean_image']['mean']-cfg['camera_offset_adu'])+cfg['effective_gaussian_sigma_adu']**2 for r in rows])
    structural_variances=np.array([r['mean_image']['std']**2 for r in rows])-noise_variances/16
    contrast_slope,_=np.polyfit(fit_frames-cfg['brightness_reference_frame'],.5*np.log(np.maximum(1,structural_variances)),1)
    cfg['log_contrast_slope_per_source_frame']=float(contrast_slope)
    cfg['brightness_model_note']='Effective brightness and contrast trends fitted on four ROIs windows0..4 only. Temporal checking uses windows5..7 at the same locations; not spatial or animal generalization.'
    write_json(ROOT/'parameters.json',cfg)
    write_json(ROOT/'calibration_search.json',records)
    print('PARAMETERS',json.dumps(cfg,indent=2),flush=True)
    return cfg

def make_masks(components,cfg):
    contribution=components*cfg['gain_adu']
    peaks=contribution.max(axis=(0,1))
    masks=contribution>=peaks[None,None,:]*cfg['mask_relative_threshold']
    for k in range(masks.shape[-1]):masks[:,:,k]=ndi.binary_fill_holes(masks[:,:,k])
    area=masks.sum(axis=(0,1))
    eligible=(peaks>=cfg['mask_min_peak_adu'])&(area>=cfg['mask_min_area_px'])&(area<=cfg['mask_max_area_px'])
    masks[:,:,~eligible]=False
    strength=np.where(masks,contribution,-1)
    ids=(strength.argmax(-1)+1).astype(np.uint16);ids[~masks.any(-1)]=0
    # Keep original cell IDs across images. Winner map resolves overlapping optical masks.
    # The full overlapping binary masks are also retained in the NPZ file.
    return ids,masks,eligible,peaks

def generate(output,seeds,per_seed,cfg):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for d in ['images','instance_masks','semantic_masks','clean','overlap_masks','diagnostics']: (output/d).mkdir()
    entries=[]
    for seed in seeds:
        fp,locs,source=load_anatomy(seed,cfg['anatomy_tag'])
        for j in range(per_seed):
            rng=np.random.default_rng(seed*10000+j)
            dx,dy=rng.uniform(-4,4,2)
            signal,components=latent(fp,cfg,rng,dx,dy)
            reference_frame=int(round(9799*j/max(1,per_seed-1)))
            brightness=float(np.exp(cfg.get('log_brightness_slope_per_source_frame',0)*(reference_frame-cfg.get('brightness_reference_frame',0))))
            contrast=float(np.exp(cfg.get('log_contrast_slope_per_source_frame',cfg.get('log_brightness_slope_per_source_frame',0))*(reference_frame-cfg.get('brightness_reference_frame',0))))
            target_signal_mean=(cfg['background_adu']+cfg['gain_adu']*signal.mean())*brightness
            components*=contrast;signal*=contrast
            background=max(0.,float(target_signal_mean-cfg['gain_adu']*signal.mean()))
            clean=cfg['camera_offset_adu']+background+cfg['gain_adu']*signal
            noisy=camera(clean,cfg,rng,16)
            ids,masks,eligible,peaks=make_masks(components,cfg)
            name=f'seed{seed}_image{j:03d}'
            tifffile.imwrite(output/'images'/f'{name}.tif',noisy[0],photometric='minisblack')
            tifffile.imwrite(output/'instance_masks'/f'{name}.tif',ids,photometric='minisblack')
            tifffile.imwrite(output/'semantic_masks'/f'{name}.tif',(ids>0).astype(np.uint8)*255,photometric='minisblack')
            tifffile.imwrite(output/'clean'/f'{name}.tif',np.minimum(clean,32767).astype(np.float32),photometric='minisblack')
            np.savez_compressed(output/'overlap_masks'/f'{name}.npz',masks=masks[:,:,eligible],cell_ids=np.flatnonzero(eligible)+1)
            tifffile.imwrite(output/'diagnostics'/f'{name}_mean16.tif',noisy.mean(0).astype(np.float32),photometric='minisblack')
            _,objects=detect(noisy.mean(0));dif=np.diff(noisy.astype(float),axis=0)/np.sqrt(2)
            noise_sd=float(np.median(np.abs(dif-np.median(dif)))/.67448975)
            entry={'name':name,'anatomy_seed':seed,'variant':j,'anatomy_file':source.relative_to(ROOT).as_posix(),
                'native_crop_center_shift_um':[float(dx),float(dy)],'instance_count':int(len(np.unique(ids))-1),
                'source_brightness_frame':reference_frame,'brightness_factor':brightness,'contrast_factor':contrast,'effective_background_adu':background,
                'single_frame':image_metrics(noisy[0]),'objects_on_16frame_mean':objects,'noise_difference_sigma_adu':noise_sd}
            entries.append(entry)
            print(f'{name}: labels={entry["instance_count"]}, detected={objects["count"]}',flush=True)
    write_json(output/'manifest.json',{'parameters':cfg,'entries':entries,'images':len(entries),
        'anatomy_seeds':list(seeds),'geometry_split_required':True,
        'source_provenance':read_json(ROOT/'reference'/'provenance.json'),
        'source_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'pipeline.py',ROOT/'sparse_scanner.py',ROOT/'generate_stage1_anatomy.m',ROOT/'measure_reference.py',ROOT/'fetch_reference.py',ROOT/'runtime_paths.py']},
        'mask_definition':'Per-cell NAOMi optical contribution>=50% own peak, holes filled, peak>=2000 ADU, area5..60px. Instance TIFF: highest contributing eligible cell wins overlaps. NPZ preserves overlaps.',
        'reference':'https://zenodo.org/records/6293696'})
    verify(output)
    return output

def verify(output):
    output=Path(output);manifest=read_json(output/'manifest.json');cfg=manifest['parameters'];ref=read_json(ROOT/'reference'/'measurements.json')
    independent_modes={};hashes={};label_counts=[];fg_fractions=[]
    for e in manifest['entries']:
        name=e['name'];im=tifffile.imread(output/'images'/f'{name}.tif');mask=tifffile.imread(output/'instance_masks'/f'{name}.tif')
        assert im.shape==mask.shape==(N,N) and im.dtype==mask.dtype==np.uint16
        assert 0<=im.min()<=im.max()<=32767 and mask.max()<=cfg['neurons_3d']
        overlap=np.load(output/'overlap_masks'/f'{name}.npz')
        assert np.array_equal(mask>0,overlap['masks'].any(-1))
        assert set(np.unique(mask))<=set(np.r_[0,overlap['cell_ids']])
        clean=tifffile.imread(output/'clean'/f'{name}.tif')
        assert np.isfinite(clean).all()
        for folder,expected in [('images',im),('instance_masks',mask),('semantic_masks',(mask>0).astype(np.uint8)*255),('clean',clean)]:
            path=output/folder/f'{name}.tif'
            with Image.open(path) as pil:
                assert np.array_equal(np.asarray(pil),expected),f'Independent Pillow read mismatch: {path}'
                independent_modes[folder]=pil.mode
            hashes[f'{folder}/{name}.tif']=hashlib.sha256(path.read_bytes()).hexdigest()
        label_counts.append(len(np.unique(mask))-1);fg_fractions.append(float((mask>0).mean()))
    write_json(output/'independent_export_validation.json',{'passed':True,'reader':'Pillow, independent of tifffile writer',
       'files_checked':len(manifest['entries'])*4,'modes':independent_modes,
       'instance_count_range':[int(min(label_counts)),int(max(label_counts))],
       'mask_foreground_fraction_range':[min(fg_fractions),max(fg_fractions)],'sha256':hashes})
    # Later windows at the four fitted real locations, and new synthetic anatomy seeds.
    real=[r for r in ref['rows'] if r['window']>=5]
    synth=[r for r in manifest['entries'] if r['anatomy_seed']!=21 and r.get('source_brightness_frame',9799)>=6988]
    if not synth:synth=[r for r in manifest['entries'] if r.get('source_brightness_frame',9799)>=6988]
    if not synth:synth=manifest['entries']
    rstats=moments([r['single_frame'] for r in real]);sstats=moments([r['single_frame'] for r in synth])
    def objstats(rows):
        a=[r['objects_on_16frame_mean'] for r in rows]
        return {'count':float(np.median([v['count'] for v in a])),
          'diameter_um':float(np.median([v['diameter_median_um'] for v in a])),
          'nn_um':float(np.median([v['nn_median_um'] for v in a]))}
    ro=objstats(real);so=objstats(synth)
    checks={}
    for k,tol in [('count',.25),('diameter_um',.20),('nn_um',.25)]:
        delta=abs(so[k]/ro[k]-1);checks[k]={'real':ro[k],'synthetic':so[k],'relative_error':delta,'tolerance':tol,'passed':bool(delta<=tol)}
    for k in ['mean','std','p10','p50','p90','gradient_rms']:
        # For brightness/quantiles compare above camera offset, not a huge shared electronics baseline.
        offset=cfg['camera_offset_adu'] if k in ['mean','p10','p50','p90'] else 0
        delta=abs((sstats[k]-offset)/(rstats[k]-offset)-1)
        checks[k]={'real':rstats[k],'synthetic':sstats[k],'relative_error_above_offset':delta,'tolerance':.25,'passed':bool(delta<=.25)}
    checks['saturation_fraction']={'real':rstats['saturation_fraction'],'synthetic':sstats['saturation_fraction'],
        'absolute_error':abs(sstats['saturation_fraction']-rstats['saturation_fraction']),'tolerance':.05,
        'passed':bool(abs(sstats['saturation_fraction']-rstats['saturation_fraction'])<=.05)}
    rn=float(np.median([r['noise_difference_sigma_adu'] for r in real]));sn=float(np.median([r['noise_difference_sigma_adu'] for r in synth]))
    checks['noise_difference_sigma_adu']={'real':rn,'synthetic':sn,'relative_error':abs(sn/rn-1),
        'tolerance':.25,'passed':bool(abs(sn/rn-1)<=.25)}
    result={'export_integrity_passed':True,'similarity_checks_passed':all(x['passed'] for x in checks.values()),
      'checks':checks,'real_check_images':len(real),'synthetic_check_images':len(synth),
      'synthetic_check_anatomies':sorted(set(x['anatomy_seed'] for x in synth)),
      'independent_anatomy_check':all(x['anatomy_seed']!=21 for x in synth),
      'limits':'One fish recording, automated morphology proxies, no expert real masks. These tolerances are engineering targets, not proof of biological realism or segmentation transfer.'}
    write_json(output/'verification.json',result)
    comparison(output,manifest,ref,result)
    print('VERIFICATION',json.dumps(result,indent=2),flush=True)
    return result

def comparison(output,manifest,ref,result):
    # Static scientific image figure: one common ADU range, no individual contrast stretching.
    font=plot_font(20)
    small=plot_font(16)
    def display(a):return Image.fromarray(np.uint8(np.clip((a-17800)/(32767-17800),0,1)*255)).convert('RGB').resize((256,256),Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(1100,1010),'white');draw=ImageDraw.Draw(canvas)
    titles=['Real single frame','NAOMi single frame','Real mean of 16 (diagnostic)','NAOMi mean of 16 (diagnostic)']
    for col,t in enumerate(titles):draw.text((15+col*275,10),t,font=small,fill='black')
    real_stack=tifffile.imread(ROOT/'reference'/'highSNR_sample.tif')
    # Deterministic unseen examples, without selecting the prettiest outputs.
    seeds=sorted(set(e['anatomy_seed'] for e in manifest['entries']))[:3]
    selected=[]
    for i,seed in enumerate(seeds):
        choices=[e for e in manifest['entries'] if e['anatomy_seed']==seed]
        selected.append(choices[min(5+i,len(choices)-1)])
    mask_canvas=Image.new('RGB',(825,55+300*len(selected)+50),'white');md=ImageDraw.Draw(mask_canvas)
    for col,t in enumerate(['NAOMi single frame','Clean image (diagnostic)','Instance mask boundaries']):md.text((15+col*275,10),t,font=small,fill='black')
    for i,e in enumerate(selected):
        roi=ref['rois'][[1,2,1][i]];win=[5,6,7][i];start=win*16
        raw=real_stack[start:start+16,roi['y']:roi['y']+N,roi['x']:roi['x']+N]
        synth=tifffile.imread(output/'images'/f'{e["name"]}.tif');mask=tifffile.imread(output/'instance_masks'/f'{e["name"]}.tif')
        overlay=np.asarray(display(synth)).copy();bound=Image.fromarray(find_boundaries(mask).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST)
        overlay[np.array(bound)>0]=[255,175,0]
        synthmean=tifffile.imread(output/'diagnostics'/f'{e["name"]}_mean16.tif')
        panels=[display(raw[0]),display(synth),display(raw.mean(0)),display(synthmean)]
        y=55+i*300
        for col,p in enumerate(panels):canvas.paste(p,(15+col*275,y))
        clean=tifffile.imread(output/'clean'/f'{e["name"]}.tif')
        for col,panel in enumerate([display(synth),display(clean),Image.fromarray(overlay)]):mask_canvas.paste(panel,(15+col*275,y))
        md.text((15,y+260),f"{e['name']}: {e['instance_count']} labeled visible contributions; ID 0 = background.",font=small,fill='black')
        draw.text((15,y+260),f"Real ROI {roi['name']}, source window {win}; synthetic seed {e['anatomy_seed']}. Shared display: 17800..32767 ADU.",font=small,fill='black')
    draw.text((15,960),'Masks come from the simulated cells. Real detection estimates are not expert annotations.',font=small,fill='black')
    canvas.save(output/'comparison.png')
    mask_canvas.save(output/'masks_preview.png')

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[21,22,23]);p.add_argument('--images-per-seed',type=int,default=8)
    p.add_argument('--calibrate-only',action='store_true');p.add_argument('--recalibrate',action='store_true');p.add_argument('--verify');p.add_argument('--output')
    args=p.parse_args()
    if args.verify:
        result=verify(args.verify)
        sys.exit(0 if result['export_integrity_passed'] and result['similarity_checks_passed'] else 2)
    if not (ROOT/'reference'/'highSNR_sample.tif').exists():
        subprocess.run([sys.executable,str(ROOT/'fetch_reference.py')],check=True)
    if not (ROOT/'reference'/'measurements.json').exists():
        subprocess.run([sys.executable,str(ROOT/'measure_reference.py')],check=True)
    cfg=calibrate() if args.recalibrate or not (ROOT/'parameters.json').exists() else read_json(ROOT/'parameters.json')
    if args.calibrate_only:return
    out=args.output or ROOT/'generated'/time.strftime('batch_%Y%m%d_%H%M%S')
    generate(out,args.seeds,args.images_per_seed,cfg)
    print(f'DONE {out}')
    if not read_json(Path(out)/'verification.json')['similarity_checks_passed']:
        print('Images were saved, but similarity targets were not all met. See verification.json.')
        sys.exit(2)

if __name__=='__main__':main()
