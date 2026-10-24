function test_naomi_scanner()
% Independent linear phantom: changing fluorescence 2 -> 3 must scale the
% clean nuclear image by 1.5, without adding the nuclear baseline twice.
naomi_octave_setup(false);
sz=[16 16 16];
vol.neur_vol=zeros(sz,'single');
indices=int32([sub2ind(sz,8,8,7),sub2ind(sz,4,4,7)]);
vol.gp_vals=cell(2,3); vol.gp_nuc=cell(2,2);
for k=1:2
    vol.gp_vals{k,1}=indices(k); vol.gp_vals{k,2}=single(1); vol.gp_vals{k,3}=true;
    vol.gp_nuc{k,1}=indices(k); vol.gp_nuc{k,2}=single(1);
end
vol.bg_proc=cell(1,2);
PSF.psf=zeros(3,3,3,'single'); PSF.psf(2,2,2)=1;
PSF.mask=ones(16,16,'single');
activity.soma=single([2 3 2;4 4 4]);
activity.dend=zeros(2,3,'single'); activity.bg=zeros(1,3,'single');
scan=struct('nuc_label',1,'motion',false,'movout',1,'scan_buff',0, ...
    'scan_avg',1,'sfrac',1,'verbose',0);
opts=struct('nt',3,'dt',0.1);
[~,clean]=scan_volume(vol,PSF,activity,scan,[],opts,[]);
first=clean(:,:,1); second=clean(:,:,2);
[delta,index]=max(second(:)-first(:));
assert(delta>0 && first(index)>0);
assert(abs(double(second(index))/double(first(index))-1.5)<1e-5);
assert(max(abs(clean(:,:,3)(:)-first(:)))<1e-5);
fprintf('PASS: nuclear scanner preserves known 2:3:2 fluorescence amplitudes\n');
end
