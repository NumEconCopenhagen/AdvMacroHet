function [vechy]=vech(y)

% Lutz Kilian
% University of Pennsylvania
% December 1993

% This function vectorizes a symmetric (q x q) matrix y by stacking the
% elements on and below the main diagonal.  The resulting vector vechy has
% dimension (q*(q+1)/2 x 1).
 
[row,column]=size(y);
if row ~= column
	'Error! This matrix is not symmetric.'
end;

vec=reshape(y,row*column,1);

vechy=vec(1:row);
for i=2:column
	vech=[vechy' vec((i-1)*row+i:i*column)']';
	vechy=vech;
end;
