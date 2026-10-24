function report=verify_naomi_output(output_dir)
% Inspect actual scientific arrays and read back exported images.
naomi_octave_setup(false);
r=load(fullfile(output_dir,'simulation_result.mat'));
assert(isequal(size(r.Fsim),size(r.Fsim_clean)));
assert(size(r.Fsim,3)==r.cfg.n_frames);
assert(all(isfinite(r.Fsim(:))) && all(isfinite(r.Fsim_clean(:))));
assert(max(abs(diff(r.time_seconds)-1/r.cfg.fps))<1e-12);
assert(sum(r.spike_counts_100Hz(:))>0,'No neural events in the recording.');
temporal_std=std(double(r.Fsim_clean),0,3);
assert(max(temporal_std(:))>1e-8,'The clean movie has no temporal activity.');
noise_mse=mean((double(r.Fsim(:))-double(r.Fsim_clean(:))).^2);
assert(noise_mse>0);
names={'activity_noisy.tif','activity_clean.tif'};
movies={r.Fsim,r.Fsim_clean};
max_decode_error=0;
for j=1:2
    info=imfinfo(fullfile(output_dir,names{j}));
    assert(numel(info)==r.cfg.n_frames);
    for k=[1,ceil(r.cfg.n_frames/2),r.cfg.n_frames]
        frame=imread(fullfile(output_dir,names{j}),k);
        assert(isa(frame,'uint16'));
        decoded=double(frame)/r.tiff_scale+r.tiff_offset;
        error=max(abs(decoded(:)-double(movies{j}(:,:,k)(:))));
        assert(error<=0.501/r.tiff_scale+1e-7);
        max_decode_error=max(max_decode_error,error);
    end
end
report=struct('passed',true,'movie_shape',size(r.Fsim), ...
    'fps',r.cfg.fps,'duration_seconds',r.cfg.n_frames/r.cfg.fps, ...
    'simulated_neurons',size(r.activity.soma,1),'neural_event_count',sum(r.spike_counts_100Hz(:)), ...
    'max_clean_temporal_std',max(temporal_std(:)),'noise_mse',noise_mse, ...
    'max_tiff_decode_error',max_decode_error,'tiff_quantization_step',1/r.tiff_scale);
fid=fopen(fullfile(output_dir,'validation.json'),'w');
fprintf(fid,'%s\n',jsonencode(report)); fclose(fid);
% A standard scientific diagnostic, rendered to disk without opening a window.
graphics_toolkit('gnuplot');
f=figure('visible','off','position',[0 0 1000 750]);
graphics_toolkit(f,'gnuplot'); % Windows CLI can otherwise create an FLTK figure.
subplot(2,2,1); imagesc(max(r.Fsim_clean,[],3)); axis image; title('Clean fluorescence, max over time');
subplot(2,2,2); imagesc(r.Fsim(:,:,ceil(r.cfg.n_frames/2))); axis image; title('Noisy fluorescence, one frame');
subplot(2,2,3); imagesc(temporal_std); axis image; title('Temporal standard deviation, clean');
subplot(2,2,4); plot(r.time_seconds,double(r.activity.soma.')); xlabel('Time, s'); ylabel('Fluorescence'); title('Known per-cell signals');
colormap(gray(256));
print(f,fullfile(output_dir,'verification.png'),'-dpng','-r120'); close(f);
disp(report);
end
