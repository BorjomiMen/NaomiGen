function root = naomi_octave_setup(build_mex)
% Load this copy's functions and packages; optionally compile native helpers.
if nargin < 1, build_mex = false; end
root = fileparts(mfilename('fullpath'));
if exist('OCTAVE_VERSION', 'builtin')
    warning('off', 'Octave:shadowed-function');
    pkg load image;
    pkg load statistics;
    pkg load signal;
    pkg load splines;
end
addpath(genpath(root));
if build_mex
    before = pwd;
    guard = onCleanup(@() cd(before));
    cd(fullfile(root, 'MEX'));
    names = {'array_SubMod','array_SubModTest','array_SubSub', ...
        'array_SubSubTest','dendrite_dijkstra_cpp','dendrite_randomwalk_cpp', ...
        'locate_neighbors'};
    for k = 1:numel(names)
        fprintf('Building %s...\n', names{k});
        mex([names{k}, '.cpp']);
    end
    rehash;
end
end
