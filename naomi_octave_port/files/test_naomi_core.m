function test_naomi_core()
naomi_octave_setup(false);
% These helpers mutate inputs. Verify values, not just compilation.
A = single([1 2 3]);
array_SubMod(A, int32([1 3]), single([10 20]));
assert(isequal(A, single([11 2 23])));
fprintf('PASS: native in-place array addition\n');
A=single([1 2 3]);
array_SubSub(A,int32([1 3]),single([10 20]));
assert(isequal(A,single([10 2 20])));
A=single([1 2 3]);
array_SubModTest(A,int32([1 3]),single([10 20]),0.5);
assert(isequal(A,single([6 2 13])));
A=single([1 2 3]);
array_SubSubTest(A,int32([1 3]),single([10 20]),0.5);
assert(isequal(A,single([5 2 10])));
fprintf('PASS: all four native array operations\n');
near=locate_neighbors(single([0 1 2]),single([0 0 0]),single([0 0 0]), ...
    single([0 0 0]),single(1.5));
assert(isequal(near,logical([1 1 0])));
points=[0 1 2;0 2 0;0 0 1];
curve=cscvn(points);
assert(max(abs(fnval(curve,curve.breaks)(:)-points(:)))<1e-12);
second=ppder(curve,2);
assert(max(abs(ppval(second,curve.breaks([1 end]))(:)))<1e-12);
line=cscvn([0 2;1 3;4 5]);
assert(max(abs(fnval(line,line.breaks)(:)-[0;1;4;2;3;5]))<1e-12);
fprintf('PASS: native neighbors and natural cubic-spline interpolation\n');
rng(23,'twister'); gp1=masked_3DGP_v2([8 8 8],[1;2],1,0);
rng(23,'twister'); gp2=masked_3DGP_v2([8 8 8],[1;2],1,0);
assert(isequal(gp1,gp2) && all(isfinite(gp1(:))) && std(gp1(:))>0);
rng(24,'twister'); gp1=masked_3DGP_test([8 8 8],[1;2],1,0);
rng(24,'twister'); gp2=masked_3DGP_test([8 8 8],[1;2],1,0);
assert(isequal(gp1,gp2) && all(isfinite(gp1(:))) && std(gp1(:))>0);
fprintf('PASS: seeded spatial Gaussian process is reproducible\n');
assert(naomi_optical_spacing([80 80],50,2,2)==40);
assert(naomi_optical_spacing([100 100],50,2,2)==50);
assert(naomi_optical_spacing([80 60],50,2,2)==20);
fprintf('PASS: optical propagation grid remains inside the FOV\n');
for count=[2 3 12 30]
    locations=rand(count,3);
    adjacency=sampSmallWorldMat([count,0],10,1,0,0,locations);
    assert(isequal(size(adjacency),[count,count]));
    assert(all(sum(adjacency,2)==min(count,10)));
end
fprintf('PASS: small-world rewiring preserves degree for small and large scenes\n');
S = zeros(3, 2000, 'single');
S(1, 600:10:650) = single(7.6e-6);
S(2, 1000:10:1050) = single(7.6e-6);
S(3, 1400:10:1450) = single(7.6e-6);
p = struct('dt',0.01,'sat_type','Ca_DE');
[CB,C,F] = calcium_dynamics(S,p,'GCaMP6s');
assert(isequal(size(F),size(S)) && all(isfinite(F(:))));
assert(all(max(F,[],2)-min(F,[],2)>0.01));
fprintf('PASS: GCaMP6s calcium dynamics, three active cells\n');
opts=struct('K',3,'nt',2000,'dt',0.01,'prot','GCaMP6s', ...
    'dyn_type','Ca_DE','dendflag',0,'axonflag',0,'N_bg',0,'p_off',0);
high=generateTimeTraces(opts,S,[],true,ones(3,1));
for fps=[5 15 30]
    opts.nt=round(20*fps); opts.dt=1/fps;
    low=generateTimeTraces(opts,S,[],true,ones(3,1));
    assert(isequal(size(low.soma),[3,opts.nt]));
    assert(all(isfinite(low.soma(:))));
    [~,hi_peak]=max(high.soma,[],2);
    [~,lo_peak]=max(low.soma,[],2);
    assert(all(abs((lo_peak-1)/fps-(hi_peak-1)/100)<=1/fps+0.011), ...
        'Resampling shifted the calcium response in physical time.');
end
fprintf('PASS: correct calcium-event timing at 5, 15, and 30 Hz\n');
short_opts=struct('K',12,'nt',17,'dt',1/15,'prot','GCaMP6s', ...
    'dyn_type','Ca_DE','dendflag',0,'axonflag',0,'N_bg',0,'p_off',0);
wrapped=generateTimeTraces(short_opts,[],rand(12,3));
assert(isequal(size(wrapped.soma),[12,17]) && all(isfinite(wrapped.soma(:))));
fprintf('PASS: automatic Hawkes wrapper and fractional-duration frame count\n');
% Independent oracle: the unmodified upstream voxelizer on the same mesh.
rng(7,'twister');
[vertices,faces]=SpiralSampleSphere(40,false);
[x,y,z]=ndgrid(linspace(-1.1,1.1,7),linspace(-1.1,1.1,6),linspace(-1.1,1.1,9));
points=[x(:),y(:),z(:)];
slow_timer=tic;
reference=octave_intriangulation_reference(vertices,faces,points);
slow_time=toc(slow_timer);
fast_timer=tic;
fast=intriangulation(vertices,faces,points);
fast_time=toc(fast_timer);
assert(isequal(reference,fast),'Optimized voxelization disagrees with upstream.');
fprintf('PASS: voxelizer matches upstream on %d points (%.2f s -> %.2f s)\n', ...
    size(points,1),slow_time,fast_time);
end
