param(
    [string]$NaomiRoot = 'C:\Users\User\NAOMi_octave'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $NaomiRoot).Path
$code = Join-Path $root 'code'
if (-not (Test-Path -LiteralPath (Join-Path $code 'VolumeCode\simulate_neural_volume.m'))) {
    throw 'Expected the complete NAOMi repository with a code directory.'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$commit = 'a3111f664faf5a7ac0ccbd6101aa07562b5e7672'
$upstream = Join-Path $root 'octave_port_upstream'
New-Item -ItemType Directory -Path $upstream -Force | Out-Null
$names = @('array_SubMod','array_SubModTest','array_SubSub','array_SubSubTest',
    'dendrite_dijkstra_cpp','dendrite_randomwalk_cpp','locate_neighbors')
foreach ($name in $names) {
    $original = Join-Path $upstream ($name + '.cpp')
    if (-not (Test-Path -LiteralPath $original)) {
        $uri = "https://api.bitbucket.org/2.0/repositories/adamshch/naomi_sim/src/$commit/code/MEX/$name.cpp"
        $body = (Invoke-WebRequest -Uri $uri -UseBasicParsing).Content
        [IO.File]::WriteAllText($original, $body, $utf8)
    }
    $body = [IO.File]::ReadAllText($original)
    $body = $body -replace '#include\s+"matrix\.h"\s*', ''
    if ($name.StartsWith('array_')) {
        $body = $body.Replace('mxGetPr(', 'mxGetData(')
    }
    if ($name -eq 'locate_neighbors') {
        $body = $body.Replace('(mxLogical*)(', '(mxLogical)(')
        $body = $body.Replace('mxGetPr(', 'mxGetData(')
        $body = $body.Replace('mxCreateLogicalArray(3,', 'mxCreateLogicalArray(mxGetNumberOfDimensions(prhs[0]),')
        $body = $body.Replace('Nx!=Ny || Nx!=Ny || Ny!=Nz', 'Nx!=Ny || Nx!=Nz || Ny!=Nz')
    }
    $body = "#include <cfloat>`n#include <climits>`n#include <algorithm>`n" + $body
    [IO.File]::WriteAllText((Join-Path $code "MEX\$name.cpp"), $body, $utf8)
}
$relative = 'TimeTraceCode\generateTimeTraces.m'
$traceOriginal = Join-Path $upstream 'generateTimeTraces.m'
if (-not (Test-Path -LiteralPath $traceOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code $relative) -Destination $traceOriginal
}
$body = [IO.File]::ReadAllText($traceOriginal)
$body = $body.Replace('spike_opts.nt = ceil(', "requested_nt = spike_opts.nt; % Preserve the requested output frame count.`nspike_opts.nt = ceil(")
$body = [regex]::Replace($body, 'n_desired\s*=\s*floor\(spike_opts.nt/\(100\*spike_opts.dt\)\);', 'n_desired = requested_nt;')
$body = [regex]::Replace($body, 'n_desired\s*=\s*spike_opts.nt;', 'n_desired = requested_nt;')
$body = [regex]::Replace($body, '(?m)^\s*S_cell\s*=\s*S_cell.bg;[^\r\n]*\r?\n\s*S_cell.bg\s*=\s*S_cell;[^\r\n]*\r?\n', '')
$body = [regex]::Replace($body, '(?s)if mod\(floor\(1\./spike_opts.dt\), 10\).*?\r?\nend', '[r2, r1] = rat(1 / (100 * spike_opts.dt), 1e-12); % Exact rational rate, including 15 Hz.')
$body = [regex]::Replace($body, 'buff\s*=\s*100;', 'buff = r1 * ceil(100 / r1);')
$body = [regex]::Replace($body, 'buff2\s*=\s*floor\(1\./spike_opts.dt\);', 'buff2 = buff * r2 / r1;')
$body = [regex]::Replace($body, '(?s)function \[r1, r2\] = pickResampleVals\(dt\).*', "function [r1, r2] = pickResampleVals(dt)`n[r2, r1] = rat(1 / (100 * dt), 1e-12);`nend`n")
[IO.File]::WriteAllText((Join-Path $code $relative), $body, $utf8)
$meshOriginal=Join-Path $upstream 'intriangulation.m'
if (-not (Test-Path -LiteralPath $meshOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code 'ExternalPackages\intriangulation.m') -Destination $meshOriginal
}
$body=[IO.File]::ReadAllText($meshOriginal)
$reference=$body.Replace('function in = intriangulation(', 'function in = octave_intriangulation_reference(')
[IO.File]::WriteAllText((Join-Path $code 'ExternalPackages\octave_intriangulation_reference.m'),$reference,$utf8)
# A voxel column repeats the same x/y ray at every z. Compute intersections
# once per distinct ray, preserving the original triangle and boundary rules.
$body=$body.Replace('for loop = 1:length(OUTPUT)', "[~, ray_first, ray_group] = unique([testx(:), testy(:)], 'rows');`nfor ray_index = 1:numel(ray_first)`n    loop = ray_first(ray_index);`n    ray_points = find(ray_group == ray_index);")
$body=$body.Replace('testz(loop)>gridCOzCROSS', 'testz(ray_points)>gridCOzCROSS')
$body=$body.Replace('testz(loop)<gridCOzCROSS', 'testz(ray_points)<gridCOzCROSS')
$body=$body.Replace('OUTPUT(loop) = voxelsINSIDE;', 'OUTPUT(ray_points) = OUTPUT(ray_points) | voxelsINSIDE;')
$body=$body.Replace('if voxelsINSIDE,break;end', '% All z positions on this ray are classified together.')
$body=$body.Replace('correctionLIST = [ correctionLIST; loop ];', 'correctionLIST = [ correctionLIST; ray_points(:) ];')
[IO.File]::WriteAllText((Join-Path $code 'ExternalPackages\intriangulation.m'),$body,$utf8)
$gpOriginal=Join-Path $upstream 'masked_3DGP_v2.m'
if (-not (Test-Path -LiteralPath $gpOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code 'MiscCode\masked_3DGP_v2.m') -Destination $gpOriginal
}
$body=[IO.File]::ReadAllText($gpOriginal)
# Use the caller's seeded stream; upstream wall-clock reseeding is unsupported
# in Octave and defeats reproducibility even when a simulation seed is set.
$body=[regex]::Replace($body, '(?m)^\s*sprev\s*=\s*rng;[^\r\n]*\r?\n','')
$body=[regex]::Replace($body, '(?m)^\s*rng\(''shuffle'',''simdTwister''\);[^\r\n]*\r?\n','')
$body=[regex]::Replace($body, '(?m)^\s*rng\(sprev\);[^\r\n]*\r?\n','')
[IO.File]::WriteAllText((Join-Path $code 'MiscCode\masked_3DGP_v2.m'),$body,$utf8)
$gpTestOriginal=Join-Path $upstream 'masked_3DGP_test.m'
if (-not (Test-Path -LiteralPath $gpTestOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code 'MiscCode\masked_3DGP_test.m') -Destination $gpTestOriginal
}
$body=[IO.File]::ReadAllText($gpTestOriginal)
$body=[regex]::Replace($body, '(?m)^\s*sprev\s*=\s*rng;[^\r\n]*\r?\n','')
$body=[regex]::Replace($body, '(?m)^\s*rng\(''shuffle'',''simdTwister''\);[^\r\n]*\r?\n','')
$body=[regex]::Replace($body, '(?m)^\s*rng\(sprev\);[^\r\n]*\r?\n','')
[IO.File]::WriteAllText((Join-Path $code 'MiscCode\masked_3DGP_test.m'),$body,$utf8)
$singleOriginal=Join-Path $upstream 'single_scan.m'
if (-not (Test-Path -LiteralPath $singleOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code 'ScanningCode\single_scan.m') -Destination $singleOriginal
}
$body=[IO.File]::ReadAllText($singleOriginal)
$body=$body.Replace('if z_sub > 1',"neur_vol2 = neur_vol; % Also define the volume when axial binning is disabled.`nif z_sub > 1")
$body=[regex]::Replace($body, '(?m)^(\s*)scan_img = ifft\(ifft\(sum', '$1scan_img = real(ifft(ifft(sum')
$body=$body.Replace("[], 1), [], 2, 'symmetric');", '[], 1), [], 2));')
[IO.File]::WriteAllText((Join-Path $code 'ScanningCode\single_scan.m'),$body,$utf8)
$blurOriginal=Join-Path $upstream 'blurredBackComp2.m'
if (-not (Test-Path -LiteralPath $blurOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code 'ScanningCode\blurredBackComp2.m') -Destination $blurOriginal
}
$body=[IO.File]::ReadAllText($blurOriginal)
$body=[regex]::Replace($body, '(?m)^(\s*)img_out = ifft\(ifft\(', '$1img_out = real(ifft(ifft(')
$body=$body.Replace("[], 1), [], 2, 'symmetric');", '[], 1), [], 2));')
[IO.File]::WriteAllText((Join-Path $code 'ScanningCode\blurredBackComp2.m'),$body,$utf8)
$lightOriginal=Join-Path $upstream 'genCorticalLightPathLite.m'
if (-not (Test-Path -LiteralPath $lightOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code 'OpticsCode\genCorticalLightPathLite.m') -Destination $lightOriginal
}
$body=[IO.File]::ReadAllText($lightOriginal)
$body=$body.Replace('psf_samp = min(psf_params.sampling,1e10);', 'psf_samp = naomi_optical_spacing(vol_sz(1:2),psf_params.sampling,vres,ss);')
[IO.File]::WriteAllText((Join-Path $code 'OpticsCode\genCorticalLightPathLite.m'),$body,$utf8)
$worldOriginal=Join-Path $upstream 'sampSmallWorldMat.m'
if (-not (Test-Path -LiteralPath $worldOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code 'MiscCode\sampSmallWorldMat.m') -Destination $worldOriginal
}
$body=[IO.File]::ReadAllText($worldOriginal)
$body=$body.Replace('if isempty(beta)', "K_conn = max(1,min(N_node,round(K_conn))); % A small scene cannot have more neighbors than nodes.`nif isempty(beta)")
$body=$body.Replace('ones(1,K_conn/2), zeros(1,N_node-K_conn/2)', 'ones(1,floor(K_conn/2)), zeros(1,N_node-floor(K_conn/2))')
$rewire=@'
for kk = 1:N_node
    connected = find(adj_mat(kk,:) == 1);
    vacant = find(adj_mat(kk,:) == 0);
    selected = find(rand(numel(connected),1) < beta);
    if numel(selected) > numel(vacant)
        selected = selected(randperm(numel(selected),numel(vacant)));
    end
    if ~isempty(selected)
        added = vacant(randperm(numel(vacant),numel(selected)));
        adj_mat(kk,connected(selected)) = 0;
        adj_mat(kk,added) = 1;
    end
end

if N_bg
'@
$body=[regex]::Replace($body,'(?s)for kk = 1:N_node[^\r\n]*\r?\n.*?\r?\nend\r?\n\r?\nif N_bg',$rewire)
[IO.File]::WriteAllText((Join-Path $code 'MiscCode\sampSmallWorldMat.m'),$body,$utf8)
$scanOriginal=Join-Path $upstream 'scan_volume.m'
if (-not (Test-Path -LiteralPath $scanOriginal)) {
    Copy-Item -LiteralPath (Join-Path $code 'ScanningCode\scan_volume.m') -Destination $scanOriginal
}
$body=[IO.File]::ReadAllText($scanOriginal)
$body=$body.Replace('nuc_act = bsxfun(@minus,nuc_act,bg_min);','nuc_act = bsxfun(@minus,nuc_act,nuc_min); % Nuclear baseline, not axonal baseline.')
[IO.File]::WriteAllText((Join-Path $code 'ScanningCode\scan_volume.m'),$body,$utf8)
Copy-Item -Path (Join-Path $PSScriptRoot 'files\*') -Destination $code -Recurse -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'START_NAOMi.cmd') -Destination (Join-Path $root 'START_NAOMi.cmd') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README_RU.md') -Destination (Join-Path $root 'README_RU.md') -Force
[IO.File]::WriteAllText((Join-Path $root 'OCTAVE_UPSTREAM_COMMIT.txt'), $commit + "`n", $utf8)
Write-Output "NAOMi Octave source prepared: $root"
