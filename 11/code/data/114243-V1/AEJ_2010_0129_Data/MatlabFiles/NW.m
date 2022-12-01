function S = NW(beta,Y,X,Z,ivopt)

% this file solves for optimal weighting matrix, correcting for
% heteroskedasticity and serial correlation, a la Newey-West.

if ivopt.linear==1
    eps=Y-X*beta;
else
    momt = fcnchk(ivopt.momt);
    f = feval(momt,beta,Y,X,Z);
    eps=Y-f;
end

T=length(X);


for i=1:rows(Y)
    M(:,i)=Z(i,:)'*eps(i);                          % moment conditions each period (instruments by T)
end

%for i=1:cols(Z)
%    for j=1:rows(Z)
%        M(i,j)=eps(j)'*Z(j,i);                      % This is vector of evaluated moments for each t
%    end
%end

z=T/(T-rows(beta));
q=ivopt.lags;                                       % Use NW weighting matrix (lags=q)
S=z*Tau(M,0);
v=1;
while v<q+1
    S=S+(1-v/(q+1))*z*(Tau(M,v)+Tau(M,v)');
    v=v+1;
end
