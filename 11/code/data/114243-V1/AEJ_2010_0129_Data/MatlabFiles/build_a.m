function [a,ia,js] = build_a(h,qcols,neq)

%  [a,ia,js] = build_a(h,qcols,neq)
%
%  Build the companion matrix, deleting inessential lags.


%  Solve for x_{t+nlead} in terms of x_{t+nlag},...,x_{t+nlead-1}.

qcols;
neqi=neq;
left  = 1:qcols;
right = qcols+1:qcols+neq;

h(:,left) = -h(:,right)\h(:,left);

%  Build the big transition matrix.

a = zeros(qcols,qcols);

if(qcols > neq)
   eyerows = 1:qcols-neq;
   eyecols = neq+1:qcols;
   a(eyerows,eyecols) = eye(qcols-neq);
end

hrows      = qcols-neq+1:qcols;
a(hrows,:) = h(:,left);

%  Delete inessential lags and build index array js.  js indexes the
%  columns in the big transition matrix that correspond to the
%  essential lags in the model.  They are the columns of q that will
%  get the unstable left eigenvectors. 

js       = 1:qcols;
zerocols = sum(abs(a)) == 0;

while( any(zerocols) )
    a(:,zerocols) = [];
    a(zerocols,:) = [];
    js(zerocols)  = [];
    zerocols = sum(abs(a)) == 0;
end

ia = length(js);

return
