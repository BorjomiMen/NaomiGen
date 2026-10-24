"""Reproducible image calibration. Detection is a proxy, not expert labels."""
from pathlib import Path
import json
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.segmentation import watershed,find_boundaries
from skimage.measure import regionprops
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
ROIS=[{'name':'A','y':112,'x':300,'split':'fit_then_temporal_check'},
      {'name':'B','y':142,'x':380,'split':'fit_then_temporal_check'},
      {'name':'C','y':234,'x':350,'split':'fit_then_temporal_check'},
      {'name':'D','y':105,'x':230,'split':'fit_then_temporal_check'}]
N=64; PIX=0.873

def detect(im):
    im=np.asarray(im,dtype=float)
    sm=ndi.gaussian_filter(im,.65)
    bg=ndi.gaussian_filter(im,4)
    response=sm-bg
    noise=np.median(np.abs(im-sm-np.median(im-sm)))/.67448975
    peaks=peak_local_max(response,min_distance=3,threshold_abs=max(300,.8*noise),exclude_border=3)
    markers=np.zeros(im.shape,np.int32)
    if len(peaks):markers[tuple(peaks.T)]=np.arange(1,len(peaks)+1)
    mask=response>max(100,.25*noise)
    seg=watershed(-sm,markers,mask=mask)
    props=[p for p in regionprops(seg) if 5<=p.area<=100 and not(p.bbox[0]==0 or p.bbox[1]==0 or p.bbox[2]==N or p.bbox[3]==N)]
    cents=np.array([p.centroid for p in props])
    diam=np.array([p.equivalent_diameter_area for p in props])*PIX
    nn=cKDTree(cents).query(cents,k=2)[0][:,1]*PIX if len(cents)>1 else np.array([])
    keep=np.isin(seg,[p.label for p in props]); seg=seg*keep
    return seg,{'count':len(props),'diameters_um':diam.tolist(),'nn_um':nn.tolist(),
        'diameter_median_um':float(np.median(diam)) if len(diam) else None,
        'nn_median_um':float(np.median(nn)) if len(nn) else None,
        'foreground_fraction':float(keep.mean())}

def image_metrics(im):
    z=np.asarray(im,dtype=float)
    return {'mean':float(z.mean()),'std':float(z.std()),
        'p10':float(np.percentile(z,10)),'p50':float(np.percentile(z,50)),
        'p90':float(np.percentile(z,90)),'saturation_fraction':float((z>=32767).mean()),
        'gradient_rms':float(np.sqrt(np.mean(np.diff(z,axis=0)**2)+np.mean(np.diff(z,axis=1)**2)))}

def main():
    ref=ROOT/'reference'; a=tifffile.imread(ref/'highSNR_sample.tif').astype(float)
    prov=json.loads((ref/'provenance.json').read_text()); groups=np.asarray(prov['window_ids'])
    rows=[]; means=[]; noisemeans=[]; noisevars=[]; images=[]
    for r in ROIS:
        stack=a[:,r['y']:r['y']+N,r['x']:r['x']+N]
        tifffile.imwrite(ref/f"roi_{r['name']}_raw.tif",stack.astype(np.int16),photometric='minisblack')
        for w in range(8):
            burst=stack[groups==w]; mean=burst.mean(0)
            seg,objects=detect(mean)
            row={**r,'window':w,'objects_on_16frame_mean':objects,'single_frame':image_metrics(burst[0]),'mean_image':image_metrics(mean)}
            rows.append(row); means.append(mean)
            d=np.diff(burst,axis=0)/np.sqrt(2)
            row['noise_difference_sigma_adu']=float(np.median(np.abs(d-np.median(d)))/.67448975)
            # Per-pixel differences remove static anatomy; robust bin variances
            for lo,hi in zip(np.arange(18000,32000,1500),np.arange(19500,33500,1500)):
                vals=d[:,(mean>=lo)&(mean<hi)].ravel()
                if vals.size>200 and w<5:
                    sigma=np.median(np.abs(vals-np.median(vals)))/.67448975
                    noisemeans.append(float(np.mean(mean[(mean>=lo)&(mean<hi)])))
                    noisevars.append(float(sigma*sigma))
            if w==0:
                lo,hi=17800,32767
                m=np.uint8(np.clip((mean-lo)/(hi-lo),0,1)*255)
                single=np.uint8(np.clip((burst[0]-lo)/(hi-lo),0,1)*255)
                rgb=np.repeat(m[...,None],3,axis=2);rgb[find_boundaries(seg)]=[255,160,0]
                panel=Image.new('RGB',(N*3,N+14));panel.paste(Image.fromarray(single),(0,14));panel.paste(Image.fromarray(m),(N,14));panel.paste(Image.fromarray(rgb),(N*2,14))
                ImageDraw.Draw(panel).text((2,0),f"{r['name']}: raw / mean16 / proxy",fill='white')
                images.append(panel.resize((768,312),Image.Resampling.NEAREST))
    overview=Image.new('RGB',(768,312*4))
    for i,p in enumerate(images):overview.paste(p,(0,i*312))
    overview.save(ref/'roi_measurements.png')
    x=np.array(noisemeans); y=np.array(noisevars)
    # Fit only non-saturated, middle-intensity bins. Constrained shot-noise variance model.
    selected=(x>18500)&(x<27000)
    slope,intercept=np.polyfit(x[selected]-17800,y[selected],1)
    coeff={'offset_adu':17800.,'variance_slope_adu':float(slope),'variance_intercept_adu2':float(max(0,intercept))}
    fit_rows=[r for r in rows if r['window']<5]
    def pool(key):return np.concatenate([np.asarray(r['objects_on_16frame_mean'][key]) for r in fit_rows])
    summary={'crop_size_px':N,'pixel_size_um':PIX,'crop_fov_um':N*PIX,
      'rois':ROIS,'fit_windows':[0,1,2,3,4],'check_windows':[5,6,7],
      'fit_proxy_count_range':[min(r['objects_on_16frame_mean']['count'] for r in fit_rows),max(r['objects_on_16frame_mean']['count'] for r in fit_rows)],
      'fit_proxy_count_median':float(np.median([r['objects_on_16frame_mean']['count'] for r in fit_rows])),
      'fit_proxy_diameter_um_q10_q50_q90':np.percentile(pool('diameters_um'),[10,50,90]).tolist(),
      'fit_proxy_nn_um_q10_q50_q90':np.percentile(pool('nn_um'),[10,50,90]).tolist(),
      'noise_model':coeff,'noise_fit_means':x.tolist(),'noise_fit_variances':y.tolist(),
      'method':'0.65px Gaussian minus 4px background; peaks distance>=3px; watershed; area5..100px; exclude border. Same proxy is used for synthetic images. Not manual ground truth.',
      'rows':rows}
    (ref/'measurements.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','noise_fit_means','noise_fit_variances']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
