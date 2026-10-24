function curve = cscvn(points)
% NAOMi subset of cscvn: centripetal natural/periodic cubic curves in pp form.
% Uses GNU Octave splines, not MATLAB Curve Fitting Toolbox. Repeated
% adjacent points split the curve into natural pieces, preserving corners.
assert(isnumeric(points) && isreal(points) && all(isfinite(points(:))));
[dimension,count]=size(points);
assert(dimension>=1 && count>=2,'cscvn needs a d-by-n array with n >= 2.');
steps=sum(diff(double(points),1,2).^2,1).^0.25;
corners=find(steps==0);
starts=[1,corners+1]; stops=[corners,count];
breaks=0; coefs=[];
for k=1:numel(starts)
    p=double(points(:,starts(k):stops(k)));
    if size(p,2)<2, continue; end
    t=[0,cumsum(sum(diff(p,1,2).^2,1).^0.25)];
    if size(p,2)==2
        part=mkpp(t,[zeros(dimension,2),(p(:,2)-p(:,1))/t(2),p(:,1)],dimension);
    elseif isempty(corners) && isequal(p(:,1),p(:,end))
        part=csape(t(:),p,'periodic');
    else
        part=csape(t(:),p,'variational');
    end
    breaks=[breaks,breaks(end)+part.breaks(2:end)];
    coefs=[coefs;part.coefs];
end
if isempty(coefs)
    curve=mkpp([0,1],[zeros(dimension,3),double(points(:,1))],dimension);
else
    curve=mkpp(breaks,coefs,dimension);
end
end
