"""Sparse evaluation of NAOMi's *same* linear scanner, checked against Octave.

Only its zero-valued voxels are omitted. PSF, axial binning, masks, tail
fluorescence, FFT padding/crop, lateral binning and units match scan_volume.
The five independent original-scanner probes are mandatory for every scene.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.io import loadmat,savemat
from scipy.signal import fftconvolve,convolve2d
from scipy.fft import rfft2,irfft2

def build_footprints(folder):
    folder=Path(folder);started=time.perf_counter()
    a=loadmat(folder/'anatomy.mat',simplify_cells=True)
    o=loadmat(folder/'optics.mat',simplify_cells=True)
    p=loadmat(folder/'scanner_probes.mat',simplify_cells=True)
    vol=a['vol'];PSF=o['PSF'];v=a['v'];shape=tuple((v['vol_sz']*v['vres']).astype(int))
    assert shape[2]==PSF['psf'].shape[2] # This route scans the full 50um volume.
    mask=np.maximum(PSF['mask'],1e-5)*PSF.get('colmask',1)
    flatmask=mask.ravel(order='F')
    kernel=PSF['psf'][:,:,::2]+PSF['psf'][:,:,1::2]
    crop=np.ceil((np.array(kernel.shape[:2])-1)/2).astype(int)
    signal_scale=float(p['scanner_sigscale']);sfrac=int(p['scan']['sfrac']);buff=int(p['scan']['scan_buff'])
    tail_size=tuple(np.asarray(p['tail_fft_size']).astype(int)[:2])
    tail_crop=np.ceil((np.array(tail_size)-np.array(shape[:2])-1)/2).astype(int)
    tails=[]
    for key in ['psfT','psfB']:
        b=PSF[key];k=b['convmask']/b['convmask'].sum()
        normmask=b['mask']/b['mask'].mean()/mask
        tails.append((rfft2(k,s=tail_size),normmask,float(b['weight'])))
    use_nuclear=p['scan'].get('nuc_label',0)>=1
    cells=vol['gp_nuc'] if use_nuclear else vol['gp_vals'];K=len(cells)
    footprints=np.zeros(((shape[0]-2*buff)//sfrac,(shape[1]-2*buff)//sfrac,K),np.float32)
    for cell in range(K):
        index1=np.asarray(cells[cell,0]).astype(np.int64).ravel()
        if not len(index1):continue
        xyz=np.array(np.unravel_index(index1-1,shape,order='F')).T
        weights=flatmask[np.maximum(1,index1%(shape[0]*shape[1]))-1]
        if not use_nuclear:weights=weights*np.asarray(cells[cell,1]).ravel()
        xmin,ymin=xyz[:,:2].min(0);xmax,ymax=xyz[:,:2].max(0)
        h,w=int(xmax-xmin+1),int(ymax-ymin+1)
        local_shape=(h+kernel.shape[0]-1,w+kernel.shape[1]-1)
        response=np.zeros(local_shape,np.float32)
        for z in np.unique(xyz[:,2]//2):
            sel=xyz[:,2]//2==z
            local=np.zeros((h,w),np.float32)
            np.add.at(local,(xyz[sel,0]-xmin,xyz[sel,1]-ymin),weights[sel])
            response+=fftconvolve(local,kernel[:,:,z],mode='full')
        direct=np.zeros((shape[0]+kernel.shape[0]-1,shape[1]+kernel.shape[1]-1),np.float32)
        direct[xmin:xmin+response.shape[0],ymin:ymin+response.shape[1]]=response
        clean=direct[crop[0]:crop[0]+shape[0],crop[1]:crop[1]+shape[1]]*(signal_scale/(2*sfrac*sfrac))
        projection=np.zeros(shape[:2],np.float32)
        np.add.at(projection,(xyz[:,0],xyz[:,1]),weights/shape[2])
        for fourier,tailmask,tailweight in tails:
            conv=irfft2(rfft2(projection*tailmask,s=tail_size)*fourier,s=tail_size)
            tail=conv[tail_crop[0]:tail_crop[0]+shape[0],tail_crop[1]:tail_crop[1]+shape[1]]
            clean+=tail*(tailweight*signal_scale/(sfrac*sfrac))
        # imgSubRowShift with motion=false has x_off=buf+1,y_off=buf+1.
        # Its historical indexing retains rows but shifts columns by one
        # native voxel. Preserve that exact offset for image/mask alignment.
        clean=clean[buff:-buff,buff+1:-buff+1]
        # MATLAB conv2(...,'same') starts at floor(kernel_size/2), including
        # even kernels; scipy convolve2d's 'same' starts one sample earlier.
        full=convolve2d(clean,np.ones((sfrac,sfrac),np.float32),mode='full')
        shift=sfrac//2
        binned=full[shift:shift+clean.shape[0],shift:shift+clean.shape[1]][::sfrac,::sfrac]
        footprints[:,:,cell]=binned
    prediction=footprints@np.asarray(p['probe_weights'])
    target=np.asarray(p['probe_clean'])
    errors=[]
    for j in range(target.shape[-1]):
        err=float(np.max(np.abs(prediction[:,:,j]-target[:,:,j]))/max(1e-15,np.max(np.abs(target[:,:,j]))))
        errors.append(err)
    report={'probe_relative_max_errors':errors,'passed':max(errors)<5e-5,
        'method':'Sparse evaluation of original NAOMi linear optics, independently compared with five original scan_volume frames',
        'seconds':time.perf_counter()-started}
    (folder/'sparse_scanner_validation.json').write_text(json.dumps(report,indent=2))
    print('SCANNER CHECK',json.dumps(report),flush=True)
    if not report['passed']:
        np.savez(folder/'scanner_debug.npz',prediction=prediction,target=target,footprints=footprints)
    assert report['passed'],f'Sparse scanner disagrees with original Octave scanner: {errors}'
    savemat(folder/'footprints.mat',{'footprints':footprints,'locs':vol['locs'],'seed':a['seed'],
        'linearity_relative_error':max(errors)},do_compression=True)
    return footprints

if __name__=='__main__':
    import sys
    build_footprints(sys.argv[1])
