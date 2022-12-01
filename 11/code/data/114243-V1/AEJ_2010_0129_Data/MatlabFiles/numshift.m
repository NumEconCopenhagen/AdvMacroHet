function [h,q,iq,nnumeric] = ...
              numshift(h,q,iq,qrows,qcols,neq,condn)

%  [h,q,iq,nnumeric] = ...
%             numshift(h,q,iq,qrows,qcols,neq,condn)
%
% Compute the numeric shiftrights and store them in q.

nnumeric = 0;
left     = 1:qcols;
right    = qcols+1:qcols+neq;

[Q,R,E]  = qr( h(:,right) );
zerorows = find( abs(diag(R)) <= condn );

while( any(zerorows) & iq <= qrows )
   h = Q'*h;
   nz = length(zerorows);
   q(iq+1:iq+nz,:) = h(zerorows,left);
   h(zerorows,:)   = shftrght( h(zerorows,:), neq );
   iq       = iq + nz;
   nnumeric = nnumeric + nz;
   [Q,R,E] = qr( h(:,right) );
   zerorows = find( abs(diag(R)) <= condn );
end

return
