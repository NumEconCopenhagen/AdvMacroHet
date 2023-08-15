function [w,rts,lgroots] = eigsys(a,uprbnd) 

%  [w,rts,lgroots] = eigsys(a,uprbnd)
%
%  Compute the roots and the left eigenvectors of the companion
%  matrix, sort the roots from large-to-small, and sort the
%  eigenvectors conformably.  Map the eigenvectors into the real
%  domain. Count the roots bigger than uprbnd.

[w,d]   = eig(a');
rts     = diag(d);
mag     = abs(rts);
[mag,k] = sort(-mag);
rts     = rts(k);
w       = w(:,k);

%  Given a complex conjugate pair of vectors W = [w1,w2], there is a
%  nonsingular matrix D such that W*D = real(W) + imag(W).  That is to
%  say, W and real(W)+imag(W) span the same subspace, which is all
%  that aim cares about. 

w = real(w) + imag(w);

lgroots = sum(abs(rts) > uprbnd);

return

