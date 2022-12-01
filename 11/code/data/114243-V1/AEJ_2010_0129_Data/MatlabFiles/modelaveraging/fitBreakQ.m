% fit.m

% this file takes univariate regression and delivers fit.

function z = fitBreakQ(Y,X,periods,shockpos,levelindex);
    
Xtemp=[X(1:(1979+.5-1969-.75)*4,:) ; X((1984-1969-.75)*4:length(X),:)];
Ytemp=[Y(1:(1979+.5-1969-.75)*4,:) ; Y((1984-1969-.75)*4:length(X),:)];

[T,K]=size(X);
beta=inv(Xtemp'*Xtemp)*(Xtemp'*Ytemp);
%res=Y-X*beta;

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

return

