function [shiftmat] = shftrght(origmat,n)

% Shifts all rows of an input matrix right n columns
% Zero out the leftmost n columns of the matrix

[rows,cols] = size(origmat);


l = (1:n);
r = (n+1:cols);
l2 = (1:cols-n);

shiftmat(:,l) = zeros(rows,n);
shiftmat(:,r) = origmat(:,l2);
    
