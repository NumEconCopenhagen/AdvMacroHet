% shocksFFR.m

% this file uses IRF of FFR to R&R shocks (irf1) to calculate needed sequence of
% shocks to generate identical IRF of FFR to alternative shocks.

% NOTE THIS SPECIFICATION ASSUMES CONTEMPORANEOUS SHOCK HAS AN EFFECT ON Y

function sh = shocksFFR(Y,X,periods,shockpos,irf1);

[T,K]=size(X);
beta=inv(X'*X)*(X'*Y);                          % coefficients
res=Y-X*beta;
beta=beta(2:length(beta));                      % drop constant
shockpos=shockpos-1;

Xa=zeros(1,length(beta)); 


sh(1)=irf1(1)/beta(shockpos);                   % this is shock in first period needed to give same IRF as in irf1
Xa(1,shockpos)=sh(1);
z(1)=Xa*beta;

for j=2:periods
    Xa(j,1)=z(j-1,1);
    for i=2:length(beta)
        Xa(j,i)=Xa(j-1,i-1);
    end

    Xa(j,shockpos)=0;
    sh(j)=(irf1(j)-Xa(j,:)*beta)/beta(shockpos); % this is shock each period needed to give same IRF as in irf1
    Xa(j,shockpos)=sh(j);
    %end
    z(j,1)=Xa(j,:)*beta;
end

z-irf1'


return

