function values = fnval(curve, points)
% Only pp-form evaluation is used by NAOMi. Other spline forms are rejected.
assert(isstruct(curve) && isfield(curve,'form') && strcmp(curve.form,'pp'), ...
    'NAOMi Octave fnval supports pp-form curves only.');
values=ppval(curve,points);
end
