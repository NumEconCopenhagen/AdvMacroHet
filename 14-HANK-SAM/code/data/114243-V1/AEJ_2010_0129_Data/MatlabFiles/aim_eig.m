function [b,rts,ia,nexact,nnumeric,lgroots,mcode] = ...
                        aim_eig(h,neq,nlag,nlead,condn,uprbnd)

                    
 %I changed b for belen.
                    
                    
%  [b,rts,ia,nexact,nnumeric,lgroots,mcode] = ...
%                       aim_eig(h,neq,nlag,nlead,condn,uprbnd)
%
%  Solve a linear perfect foresight model using the matlab eig
%  function to find the invariant subspace associated with the big
%  roots.  This procedure will fail if the companion matrix is
%  defective and does not have a linearly independent set of
%  eigenvectors associated with the big roots.
% 
%  Input arguments:
% 
%    h         Structural coefficient matrix (neq,neq*(nlag+1+nlead)).
%    neq       Number of equations.
%    nlag      Number of lags.
%    nlead     Number of leads.
%    condn     Zero tolerance used as a condition number test
%              by numeric_shift and reduced_form.
%    uprbnd    Inclusive upper bound for the modulus of roots
%              allowed in the reduced form.
% 
%  Output arguments:
% 
%    b         Reduced form coefficient matrix (neq,neq*nlag).
%    rts       Roots returned by eig.
%    ia        Dimension of companion matrix (number of non-trivial
%              elements in rts).
%    nexact    Number of exact shiftrights.
%    nnumeric  Number of numeric shiftrights.
%    lgroots   Number of roots greater in modulus than uprbnd.
%    mcode     Return code: see function aimerr.

if(nlag<1 | nlead<1) 
    error('Aim_eig: model must have at least one lag and one lead.');
end

% Initialization.

nexact   = 0;
nnumeric = 0;
lgroots  = 0;
iq       = 0;
mcode    = 0;

qrows = neq*nlead;
qcols = neq*(nlag+nlead);
bcols = neq*nlag;
 
q        = zeros(qrows,qcols);
rts      = zeros(qcols,1);

% Compute the auxiliary initial conditions and store them in q.

[h,q,iq,nexact] = ex_shift(h,q,iq,qrows,qcols,neq);

   if (iq>qrows) 
      mcode = 61;
      return;
   end

[h,q,iq,nnumeric] = numshift(h,q,iq,qrows,qcols,neq,condn);

   if (iq>qrows) 
      mcode = 62;
      return;
   end

%  Build the companion matrix.  Compute the stability conditions, and
%  combine them with the auxiliary initial conditions in q.  

[a,ia,js] = build_a(h,qcols,neq);

if (ia ~= 0)

   [w,rts,lgroots] = eigsys(a,uprbnd);

   q = copy_w(q,w,js,iq,qrows);

end

   test = nexact+nnumeric+lgroots;
       if (test > qrows) mcode = 3;
   elseif (test < qrows) mcode = 4;
   end

% If the right-hand block of q is invertible, compute the reduced form.

[nonsing,b] = reduform(q,qrows,qcols,bcols,neq,condn);

%[nonsing,b] = reduform(q,qrows,qcols,bcols,neq,zeros(neq,neq*nlag),condn);

    if ( nonsing & mcode==0) mcode =  1;
elseif (~nonsing & mcode==0) mcode =  5;
elseif (~nonsing & mcode==3) mcode = 35;
elseif (~nonsing & mcode==4) mcode = 45;
end

return
