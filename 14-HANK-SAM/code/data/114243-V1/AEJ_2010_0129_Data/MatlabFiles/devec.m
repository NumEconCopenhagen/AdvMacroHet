% ============================================================ %
% devec.m
%
% For a vector (m*n,1), returns A (m,n) (see Hamilton p. 265)
% ------NOTE: pass the function the number of rows in the matrix A!!!
% 
% code: --> A = devec(v,m)
% ============================================================ % 

function A = devec(v,m);
  l = length(v);
  n = l/m;
  A = [];
  for i = 1:n;
    A = [A, v(((i-1)*m)+1:i*m)];
  end;
