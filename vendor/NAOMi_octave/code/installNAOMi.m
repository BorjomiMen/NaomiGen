% Portable NAOMi initialization. Detect this runtime's native extension.
naomi_octave_setup(false);
naomi_mex_names={'array_SubMod','array_SubModTest','array_SubSub', ...
    'array_SubSubTest','dendrite_dijkstra_cpp','dendrite_randomwalk_cpp','locate_neighbors'};
naomi_build_required=false;
for naomi_mex_index=1:numel(naomi_mex_names)
    if exist(naomi_mex_names{naomi_mex_index},'file')~=3
        naomi_build_required=true;
    end
end
if naomi_build_required, naomi_octave_setup(true); end
clear naomi_mex_names naomi_mex_index naomi_build_required;
