function scof = obstruct(cof,cofb,neq,nlag,nlead)

%--------------------------------------------------------
%
% construct the coefficients in the observable structure.
%    
%   inputs:  
%            cof    structural coefficients
%            cofb   reduced form
%            neq
%            nlag
%            nlead
%   output:
%            scof  observable structure coefficients
%    
%--------------------------------------------------------

% append negative identity to cofb

cofb = [cofb -eye(neq)];

scof = zeros(neq,neq*(nlag+1));
q = zeros(neq*nlead, neq*(nlag+nlead));

[rc,cc] = size(cofb);
q(1:rc,1:cc) = cofb;

if( nlead > 1 ) 

   for i = 1:(nlead-1)
      rows = i*neq + (1:neq);
      q(rows,:) = shftrght( q((rows-neq),:), neq );
   end

end

l = (1: neq*nlag);
r = (neq*nlag+1: neq*(nlag+nlead));
q(:,l) = -q(:,r) \ q(:,l);

minus = (             1:       neq*(nlag+1));
plus  = (neq*(nlag+1)+1: neq*(nlag+1+nlead));

scof(:,neq+1:neq*(nlag+1)) = cof(:,plus)*q(:,l);
scof = scof + cof(:,minus);

return

