% impresp.m

% this file takes VAR coefficients, returns response of all variables to
% Choleski identified shock to variable FFRpos.  Assume Beta includes
% constant coefficient.  

function z = impresp(Beta,A,FFRpos,K,periods,u);

Xa=zeros(1,length(Beta)-1);     % ignore constant
Ya=zeros(periods,K);
ua=zeros(1,K);  ua(1,FFRpos)=std(u(:,FFRpos));
ua=(A*ua')';
Ainv=inv(A);
VARlags=(length(Beta)-1)/K;
Ya(1,:)=(Ainv*ua')';
for j=2:periods
    Xa(j,2:K*VARlags)=Xa(j-1,1:K*VARlags-1);
    for k=1:K
        Xa(j,VARlags*(k-1)+1)=Ya(j-1,k);
        Ya(j,:)=Xa(j,:)*Beta(2:length(Beta),:);
    end
end

z=Ya(:,:);

return

