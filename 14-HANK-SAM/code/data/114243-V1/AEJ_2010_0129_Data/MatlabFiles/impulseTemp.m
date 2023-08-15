% impulseTemp.m

% this file takes beta and constructs impulse response to a
% one-unit innovation in exogenous shock.  If levelindex=1, returns irf of
% level, if =0, returns cumulative irf's.

% NOTE THIS SPECIFICATION ASSUMES CONTEMPORANEOUS SHOCK HAS NO EFFECT ON Y

function irf = impulseTemp(beta,periods,shockpos,levelindex);


Xa=zeros(1,length(beta)-1); %Xa(shockpos-1)=1;   % drop constant
%z(1)=Xa*beta(2:length(beta));
z(1)=0;

for j=2:periods
    Xa(j,1)=z(j-1,1);
    for i=2:length(beta)-1
        Xa(j,i)=Xa(j-1,i-1);
    end
    if j==2
        Xa(j,shockpos-1)=1;
    else
        Xa(j,shockpos-1)=0;
    end
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

