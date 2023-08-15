% impulsesh.m

% this file takes Y, regresses on X.  Then, assuming X is ordered with
% constant, then lags of X, then lags of exogenous shock (shockpos
% indicates when shock lags enter X), constructs impulse response to a
% one-unit innovation in exogenous shock.  If levelindex=1, returns irf of
% level, if =0, returns cumulative irf's.

% NOTE THIS SPECIFICATION ASSUMES ONLY LAGGED SHOCK HAS AN EFFECT ON Y

% THIS SPECIFICATION FEEDS SHOCK SEQUENCE SPECIFIED IN shock

function irf = impulsesh(Y,X,periods,shockpos,levelindex,shock);

[T,K]=size(X);
beta=inv(X'*X)*(X'*Y);
res=Y-X*beta;

Xa=zeros(1,length(beta)-1); %Xa(shockpos-1)=shock(1);   % drop constant
%z(1)=Xa*beta(2:length(beta));
z(1)=0;

for j=2:periods
    Xa(j,1)=z(j-1,1);
    for i=2:length(beta)-1
        Xa(j,i)=Xa(j-1,i-1);
    end
 %   if j==2
  %      Xa(j,shockpos-1)=1;
   % else
        Xa(j,shockpos-1)=shock(j-1);
    %end
    z(j,1)=Xa(j,:)*beta(2:length(beta));
end

if levelindex==0
    irf(1)=z(1);
    for j=2:length(z)
        irf(j)=irf(j-1)+z(j);
    end
else
    irf=z';
end

return

