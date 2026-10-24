function output = generate_stage1_anatomy(seed, cfg)
% Original NAOMi morphology, voxelization, optical propagation, and scanner.
% Image-only task: no dendrite growth, temporal activity, or learned refiner.
if nargin<1, seed=21; end
if nargin<2, cfg=struct(); end
here=fileparts(mfilename('fullpath'));
addpath('C:/Users/User/NAOMi_octave/code');
naomi_octave_setup(false);
defaults=struct('volume_um',[80 80 50],'neurons',600,'radius_um',2.2, ...
 'nucleus_radius_um',1.5,'min_distance_um',5.0,'tag','v5','depth_um',30, ...
 'cell_slab_um',20,'fluorescence_mode','soma','nuclear_fill_fraction',.1);
names=fieldnames(defaults);
for j=1:numel(names)
 if ~isfield(cfg,names{j}),cfg.(names{j})=defaults.(names{j});end
end
output=fullfile(here,'anatomy',sprintf('%s_seed%d',cfg.tag,seed));
if ~exist(output,'dir'),mkdir(output);end
diary(fullfile(output,'run.log')); guard=onCleanup(@() diary('off'));
rng(seed,'twister'); alltime=tic;
fprintf('Stage1 image anatomy: %s, seed%d\n',cfg.tag,seed);
v=check_vol_params(struct('vol_sz',cfg.volume_um,'N_neur',cfg.neurons, ...
 'min_dist',cfg.min_distance_um,'N_den',0,'vol_depth',cfg.depth_um,'verbose',1));
n=check_neur_params(struct('neur_type','sphere','avg_rad',cfg.radius_um, ...
 'nuc_rad',[cfg.nucleus_radius_um 1],'p_scale',15*(cfg.radius_um/2.7)^2,'l_scale',90, ...
 'exts',[0.85 1.15],'eccen',[.25 .25 .35],'min_thic',[.3 .2], ...
 'n_samps',100,'nuc_fluorsc',0));
psf=check_psf_params([]);psf.fastmask=false;
% The upstream tail interpolation assumes the same available propagation
% distance above and below the focal volume. Do not request 50um above a
% volume whose surface depth is only 30um.
psf.taillength=min(psf.taillength,v.vol_depth);
psf.sampling=naomi_optical_spacing(v.vol_sz(1:2),50,v.vres,psf.ss);
v.vasc_sz=gaussianBeamSize(psf,v.vol_depth+v.vol_sz(3)/2)+v.vol_sz+[0 0 v.vol_depth];
anatomy_file=fullfile(output,'anatomy.mat');
if exist(anatomy_file,'file')
 load(anatomy_file,'vol','v','n','anatomy_rng');rng(anatomy_rng);
else
 stage=tic;
 % Use the exact upstream soma generation stages. The image-only nuclear
 % mode does not require growing invisible pyramidal dendritic arbors.
 vessels=false(v.vres*(v.vol_sz+[0 0 v.vol_depth]));
 bank=fullfile(here,'anatomy',sprintf('v1_seed%d',seed),'anatomy.mat');
 if strcmp(cfg.tag,'v2') && exist(bank,'file')
  old=load(bank,'vol');locs=old.vol.locs;Vcell=old.vol.Vcell;
  Tri=old.vol.Tri;rotAng=old.vol.rotAng;
 elseif strcmp(cfg.tag,'v5') && exist(fullfile(here,'anatomy',sprintf('v4_seed%d',seed),'anatomy.mat'),'file')
  old=load(fullfile(here,'anatomy',sprintf('v4_seed%d',seed),'anatomy.mat'),'vol','cfg','anatomy_rng');
  locs=old.vol.locs;Vcell=old.vol.Vcell;Tri=old.vol.Tri;rotAng=old.vol.rotAng;
  for k=1:size(Vcell,3)
   Vcell(:,:,k)=single(bsxfun(@plus,bsxfun(@minus,double(Vcell(:,:,k)),double(locs(k,:)))*(cfg.radius_um/old.cfg.radius_um),double(locs(k,:))));
  end
  rng(old.anatomy_rng);
 else
  placement=v;placement.vol_sz(3)=cfg.cell_slab_um;
  [locs,Vcell,Vnuc,Tri,rotAng]=sampleDenseNeurons(n,placement,vessels);
  shift=(v.vol_sz(3)-placement.vol_sz(3))/2;
  locs(:,3)=locs(:,3)+shift;Vcell(:,3,:)=Vcell(:,3,:)+shift;
 end
 % The mouse model's fixed 3um nucleus offset and origin-based rescaling
 % are inappropriate for 3um fish cells. Reuse NAOMi surface shapes, but
 % center the nuclear mesh inside its soma and scale around that center.
 Vnuc=zeros(size(Vcell),'single');
 for k=1:size(Vcell,3)
  centered=bsxfun(@minus,double(Vcell(:,:,k)),double(locs(k,:)));
  [~,cell_volume]=convhull(centered(:,1),centered(:,2),centered(:,3));
  factor=((4*pi/3)*cfg.nucleus_radius_um^3/cell_volume)^(1/3);
  assert(factor<1,'The nucleus must fit inside the cell body.');
  Vnuc(:,:,k)=single(bsxfun(@plus,centered*factor,double(locs(k,:))));
 end
 v.N_neur=size(Vcell,3);
 [soma_idx,fluor,nuc,soma]=generateNeuralVolume(n,v,locs,Vcell,Vnuc,vessels);
 assert(sum(cellfun(@numel,nuc(:,1))==0)<.03*size(nuc,1),'Too many empty nuclei.');
 K=v.N_neur; gp=cell(K,3);
 for k=1:K
  gp{k,1}=[soma{k};nuc{k,1}];
  gp{k,2}=[ones(numel(soma{k}),1,'single'); ...
   cfg.nuclear_fill_fraction*ones(numel(nuc{k,1}),1,'single')];
  gp{k,3}=true(numel(gp{k,1}),1);
 end
 vol=struct('neur_vol',fluor,'gp_nuc',{nuc},'gp_soma',{soma},'gp_vals',{gp}, ...
 'bg_proc',{cell(0,2)},'locs',locs,'Vcell',Vcell,'Vnuc',Vnuc,'Tri',Tri,'rotAng',rotAng, ...
 'neur_ves',vessels,'neur_ves_all',false(v.vres*v.vasc_sz));
 anatomy_rng=rng;
 save('-mat7-binary',anatomy_file,'vol','v','n','anatomy_rng','cfg','seed');
 fprintf('Morphology %.2fs, actual K=%d\n',toc(stage),K);
end
% NAOMi's somatic MEX writes voxel values instead of summing them. Assign
% every fluorescent voxel to one cell in advance, including the few
% nucleus/soma collisions in dense tissue, so optics remains linear.
owner=zeros(size(vol.neur_vol),'uint16');
for k=size(vol.gp_vals,1):-1:1
 indices=vol.gp_vals{k,1};keep=(owner(indices)==0);
 vol.gp_vals{k,1}=indices(keep);
 vol.gp_vals{k,2}=vol.gp_vals{k,2}(keep);
 vol.gp_vals{k,3}=vol.gp_vals{k,3}(keep);
 owner(vol.gp_vals{k,1})=k;
end
save('-mat7-binary',anatomy_file,'vol','v','n','anatomy_rng','cfg','seed');
optics_file=fullfile(output,'optics.mat');
if exist(optics_file,'file')
 load(optics_file,'PSF','optics_rng');rng(optics_rng);
else
 stage=tic;PSF=simulate_optical_propagation(v,psf,vol);optics_rng=rng;
 save('-mat7-binary',optics_file,'PSF','psf','optics_rng');
 fprintf('Original optical propagation %.2fs\n',toc(stage));
end
stage=tic;
K=size(vol.gp_vals,1);
locs=vol.locs;
vol.bg_proc=cell(1,2); % Upstream scanner expects one empty background slot.
% Save independent original-scanner probes. Python evaluates the same
% linear optical convolution for all cells using sparse local supports.
scan=check_scan_params(struct('nuc_label',strcmp(cfg.fluorescence_mode,'nuclear'),'motion',false,'movout',1, ...
 'scan_buff',10,'sfrac',2,'verbose',1));
noise=check_noise_params(struct('bleedp',0));
opts=check_spike_opts(struct('K',K,'nt',5,'dt',1/15));
tpm=check_tpm_params([]);
probe_weights=zeros(K,5,'single');probe_weights(:,1)=1;
[~,order]=sort(sum(bsxfun(@minus,double(locs),[40 40 25]).^2,2));
probe_weights(order(1),3)=1;probe_weights(order(2),4)=1;
probe_weights(:,5)=single(.2+rand(K,1));
activity=struct('soma',probe_weights,'dend',zeros(K,5,'single'),'bg',zeros(1,5,'single'));
[ignored,probe_clean]=scan_volume(vol,PSF,activity,scan,noise,opts,tpm);clear ignored;
scanner_sigscale=tpmSignalscale(tpm)*opts.dt*scan.sfrac^2/(250*250);
tail_fft_size=size(psf_fft(size(vol.neur_vol),PSF.psfT.convmask));
save('-mat7-binary',fullfile(output,'scanner_probes.mat'),'probe_weights','probe_clean', ...
 'scanner_sigscale','tail_fft_size','cfg','seed','v','n','scan','psf','locs');
fprintf('Original-scanner probes %.2fs\n',toc(stage));
metadata=struct('seed',seed,'cfg',cfg,'actual_cells',K, ...
 'native_pixel_um',scan.sfrac/v.vres, ...
 'total_seconds',toc(alltime),'stages','Original NAOMi sampleDenseNeurons, generateNeuralVolume, simulate_optical_propagation, scan_volume', ...
 'nuclear_mode','filled small fluorescent bodies; chosen image morphology, not a verified transgenic line claim');
fid=fopen(fullfile(output,'metadata.json'),'w');fprintf(fid,'%s\n',jsonencode(metadata));fclose(fid);
fprintf('DONE %s (%.1fs)\n',output,toc(alltime));
end
