function spacing=naomi_optical_spacing(lateral_um,requested_um,vres,oversampling)
% Coarse propagation nodes must end at the actual FOV, not beyond it.
pixels=lateral_um*vres*oversampling;
assert(all(abs(pixels-round(pixels))<1e-7),'Optical FOV must align to the propagation pixel grid.');
common=gcd(round(pixels(1)),round(pixels(2)));
largest=min(common,floor(requested_um*vres*oversampling));
assert(largest>=1,'Requested optical sampling is smaller than one propagation pixel.');
candidates=1:largest;
step_px=max(candidates(mod(common,candidates)==0));
spacing=step_px/(vres*oversampling);
end
