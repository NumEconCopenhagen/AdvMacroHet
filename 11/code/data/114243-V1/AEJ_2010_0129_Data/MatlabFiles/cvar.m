% ============================================================== %
% cvar.m 
% finds the variance/covariance matrix of the VAR:
% 		y(t) = Ay(t-1) + Be(t)
% the program takes as inputs, A,B,V(e) and returns V(y).
% -- (these are amat, b, and shockvar in AIM)
%
% ( V(y) = AV(y)A' + BV(e)B' )... see Hamilton p. 265.
%
% written by C. House 10/05/00
% ============================================================== %

function V = cvar(A,B,V_e);

  l1 = length(A);
  Q  = B*V_e*B';
  [m,n]=size(Q)
  vQ = vec(Q);
  [tt,cc]=size(vQ);
  AA = kron(A,A);
  l  = length(AA);
  [oo,pp]=size(inv(eye(l)-AA));
  vV = inv(eye(l)-AA)*vQ;
  V  = devec(vV,l1);



