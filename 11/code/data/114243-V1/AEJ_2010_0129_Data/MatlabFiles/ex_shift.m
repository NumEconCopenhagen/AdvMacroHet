function [h,q,iq,nexact] = ex_shift(h,q,iq,qrows,qcols,neq)

% [h,q,iq,nexact] = ex_shift(h,q,iq,qrows,qcols,neq)
%
% Compute the exact shiftrights and store them in q.

nexact = 0;
left   = 1:qcols;
qcols;
right  = qcols+1:qcols+neq;

zerorows = find( sum(abs( h(:,right)' ))==0 );

while( any(zerorows) & iq <= qrows )

   nz = length(zerorows);

   q(iq+1:iq+nz,:) = h(zerorows,left);
   h(zerorows,:)   = shftrght(h(zerorows,:),neq);

   iq     = iq + nz;
   nexact = nexact + nz;

   zerorows = find( sum(abs( h(:,right)' ))==0 );

end

return
