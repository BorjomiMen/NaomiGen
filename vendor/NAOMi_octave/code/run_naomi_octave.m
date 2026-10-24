function output_dir = run_naomi_octave(cfg)
% Small original NAOMi nuclear two-photon pipeline, with portable exports.
root = naomi_octave_setup(false);
if nargin < 1, cfg = struct(); end
defaults = struct('seed',1,'volume_um',[80 80 50], 'depth_um',50, ...
    'neurons',12,'fps',15,'n_frames',300,'output_root', ...
    fullfile(fileparts(root),'results'),'run_name','','export_avi',true,'resume',false, ...
    'refine_optical_mask',false);
keys = fieldnames(defaults);
for k=1:numel(keys)
    if ~isfield(cfg,keys{k}), cfg.(keys{k})=defaults.(keys{k}); end
end
assert(cfg.n_frames>=2 && cfg.n_frames==fix(cfg.n_frames));
assert(cfg.fps>0 && cfg.fps<=100 && cfg.neurons>=2);
rng(cfg.seed,'twister');
if isempty(cfg.run_name)
    cfg.run_name=sprintf('nuclear_%s_seed%d',datestr(now,'yyyymmdd_HHMMSS'),cfg.seed);
end
output_dir=fullfile(cfg.output_root,cfg.run_name);
assert(~exist(output_dir,'dir') || cfg.resume,'Output directory already exists; choose a new run_name.');
if ~exist(output_dir,'dir'), mkdir(output_dir); end
diary(fullfile(output_dir,'run.log'));
log_guard=onCleanup(@() diary('off'));
total_timer=tic;
fprintf('Octave %s; seed %d; %g fps; %d frames\n',version,cfg.seed,cfg.fps,cfg.n_frames);
v=check_vol_params(struct('vol_sz',cfg.volume_um,'vol_depth',cfg.depth_um, ...
    'N_neur',cfg.neurons,'N_den',0,'verbose',1));
n=check_neur_params([]);
vasc=check_vasc_params([]); vasc.flag=0;
d=check_dend_params([]);
bg=check_bg_params([]); bg.flag=0;
ax=check_axon_params([]); ax.flag=0;
psf=check_psf_params([]);
psf.fastmask=cfg.refine_optical_mask;
psf.sampling=naomi_optical_spacing(cfg.volume_um(1:2),psf.sampling,v.vres,psf.ss);
scan=check_scan_params([]); scan.nuc_label=1; scan.motion=false; scan.movout=1;
noise=check_noise_params([]);
tpm=check_tpm_params([]);
fprintf('[1/5] Original NAOMi anatomy\n');
stage=tic;
anatomy_file=fullfile(output_dir,'anatomy_checkpoint.mat');
if cfg.resume && exist(anatomy_file,'file')
    load(anatomy_file,'vol','v','n','vasc','d','bg','ax','anatomy_rng','anatomy_cfg');
    assert(isequal(anatomy_cfg.volume_um,cfg.volume_um) && anatomy_cfg.depth_um==cfg.depth_um ...
        && anatomy_cfg.neurons==cfg.neurons && anatomy_cfg.seed==cfg.seed,'Checkpoint anatomy differs from requested configuration.');
    rng(anatomy_rng);
else
    [vol,v,n,vasc,d,bg,ax]=simulate_neural_volume(v,n,vasc,d,bg,ax,psf);
    anatomy_rng=rng; anatomy_cfg=cfg;
    save('-mat7-binary',anatomy_file,'vol','v','n','vasc','d','bg','ax','anatomy_rng','anatomy_cfg');
end
timings.anatomy=toc(stage);
fprintf('Anatomy %.2f s\n',timings.anatomy);
fprintf('[2/5] Original NAOMi optical propagation\n'); stage=tic;
optics_file=fullfile(output_dir,'optics_checkpoint.mat');
if cfg.resume && exist(optics_file,'file')
    load(optics_file,'PSF','optics_rng');
    rng(optics_rng);
else
    PSF=simulate_optical_propagation(v,psf,vol);
    optics_rng=rng;
    save('-mat7-binary',optics_file,'PSF','optics_rng');
end
timings.optics=toc(stage);
fprintf('Optics %.2f s\n',timings.optics);
fprintf('[3/5] Activity and GCaMP6s dynamics\n'); stage=tic;
K=size(vol.gp_vals,1);
assert(K>=2,'At least two cells must fit in the generated volume.');
opts=check_spike_opts(struct('K',K,'nt',cfg.n_frames,'dt',1/cfg.fps, ...
    'prot','GCaMP6s','dyn_type','Ca_DE','rate',1,'p_off',0, ...
    'N_bg',0,'dendflag',0,'axonflag',0));
internal_dt=0.01;
internal_count=ceil(cfg.n_frames*100/cfg.fps);
event_opts=opts; event_opts.dt=internal_dt; event_opts.nt=internal_count;
[events,network]=genCorrelatedSpikeTrains2(event_opts,[],vol.locs,true,false);
spike_counts_100Hz=single(events.soma);
spike_predef=single(7.6e-6)*spike_counts_100Hz;
[activity,spikes,expression_factors]=generateTimeTraces(opts,spike_predef,vol.locs);
activity.dend=zeros(K,cfg.n_frames,'single');
if isempty(vol.bg_proc), vol.bg_proc=cell(1,2); end
activity.bg=zeros(size(vol.bg_proc,1),cfg.n_frames,'single');
assert(isequal(size(activity.soma),[K,cfg.n_frames]));
timings.activity=toc(stage);
fprintf('[4/5] Original NAOMi scanner\n'); stage=tic;
noise.sigscale=tpmSignalscale(tpm)*opts.dt*scan.sfrac^2/(250*250);
[Fsim,Fsim_clean]=scan_volume(vol,PSF,activity,scan,noise,opts,tpm);
timings.scanning=toc(stage);
assert(isequal(size(Fsim),size(Fsim_clean)) && size(Fsim,3)==cfg.n_frames);
assert(all(isfinite(Fsim(:))) && all(isfinite(Fsim_clean(:))));
assert(max(Fsim_clean(:))>min(Fsim_clean(:)),'The clean movie is empty/constant.');
fprintf('[5/5] Saving paired data and ground truth\n'); stage=tic;
time_seconds=(0:cfg.n_frames-1)/cfg.fps;
metadata=struct('engine','Original NAOMi MATLAB source, Octave compatibility port', ...
    'modality','nuclear two-photon GCaMP6s','calibrated_to_zebrafish',false, ...
    'internal_dt_seconds',internal_dt,'octave_version',version, ...
    'note','Small synthetic baseline. Not calibrated light-sheet or zebrafish anatomy.');
metadata.pixel_size_um=scan.sfrac/v.vres;
metadata.nuclear_baseline_corrected=true;
metadata.random_generator='twister, seeded globally; no wall-clock reseeding';
metadata.optical_mask_refinement=cfg.refine_optical_mask;
metadata.coarse_optical_spacing_um=psf.sampling;
tiff_offset=double(min([Fsim(:);Fsim_clean(:)]));
tiff_scale=65535/max(eps,double(max([Fsim(:);Fsim_clean(:)]))-tiff_offset);
save('-mat7-binary',fullfile(output_dir,'simulation_result.mat'), ...
    'Fsim','Fsim_clean','activity','spikes','spike_counts_100Hz','expression_factors', ...
    'time_seconds','cfg','metadata','vol','PSF','network','v','n','scan','noise', ...
    'psf','tpm','tiff_offset','tiff_scale');
naomi_write_stack(Fsim,fullfile(output_dir,'activity_noisy.tif'),tiff_offset,tiff_scale);
naomi_write_stack(Fsim_clean,fullfile(output_dir,'activity_clean.tif'),tiff_offset,tiff_scale);
preview=uint16(round((double(max(Fsim_clean,[],3))-tiff_offset)*tiff_scale));
imwrite(preview,fullfile(output_dir,'preview.png'));
if cfg.export_avi
    pkg load video;
    writer=VideoWriter(fullfile(output_dir,'activity_noisy.avi'));
    writer.FrameRate=cfg.fps;
    open(writer);
    for k=1:cfg.n_frames
        gray=uint8(round((double(Fsim(:,:,k))-tiff_offset)*tiff_scale/257));
        writeVideo(writer,repmat(gray,[1 1 3]));
    end
    close(writer);
end
timings.export=toc(stage); timings.total=toc(total_timer);
fid=fopen(fullfile(output_dir,'metadata.json'),'w');
fprintf(fid,'%s\n',jsonencode(struct('cfg',cfg,'metadata',metadata,'timings',timings, ...
    'movie_shape',size(Fsim),'neurons',K)));
fclose(fid);
fprintf('DONE %s\nTotal %.2f s\n',output_dir,timings.total);
end
