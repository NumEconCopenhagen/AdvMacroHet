% fit.m

% this file takes univariate regression and delivers fit.

function z = fit1(Y,X,periods,shockpos,levelindex);

[T,K]=size(X);
beta=inv(X'*X)*(X'*Y);
res=Y-X*beta;

Xa=X(1,:);
Y1=Xa*beta;

for j=2:T
    Xa(j,1)=1;
    Xa(j,2)=Y1(j-1,1);
    for i=3:length(beta)
        Xa(j,i)=Xa(j-1,i-1);
    end
    Xa(j,shockpos)=X(j,shockpos);
    Y1(j,1)=Xa(j,:)*beta;
end

if levelindex==0
    z(1)=Y(1);
    for j=2:length(Y1)
        z(j)=z(j-1)+Y1(j);
    end
else
    z=Y1;
end

if levelindex==0
    z=100*(z(13:length(z))-z(1:length(z)-12));
end

return

