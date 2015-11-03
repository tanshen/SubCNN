function [o, o2] = boxoverlap(a, b)
% Compute the symmetric intersection over union overlap between a set of
% bounding boxes in a and a single bounding box in b.
%
% a  a matrix where each row specifies a bounding box
% b  a single bounding box

x1 = max(a(:,1), b(1));
y1 = max(a(:,2), b(2));
x2 = min(a(:,3), b(3));
y2 = min(a(:,4), b(4));

% w = x2-x1+1;
% h = y2-y1+1;
% inter = w.*h;
% aarea = (a(:,3)-a(:,1)+1) .* (a(:,4)-a(:,2)+1);
% barea = (b(3)-b(1)+1) * (b(4)-b(2)+1);

w = x2-x1;
h = y2-y1;
inter = w.*h;
aarea = (a(:,3)-a(:,1)) .* (a(:,4)-a(:,2));
barea = (b(3)-b(1)) * (b(4)-b(2));

% intersection over union overlap
o = inter ./ (aarea+barea-inter);
% set invalid entries to 0 overlap
o(w <= 0) = 0;
o(h <= 0) = 0;

if nargout == 2
    o2 = inter ./ aarea;
    % set invalid entries to 0 overlap
    o2(w <= 0) = 0;
    o2(h <= 0) = 0;
end
