%%%  This file takes parameters from an AR(2) process and converts them
%%%  into AR(2) parameters of lower frequency process.

function rhoa = averaging(rho,N,its);
rho1=rho(1);
rho2=rho(2);
%N=3;
%its=20000;

MM=[rho1 rho2; 1 0];
if max(abs(eig(MM)))>1
    flag0=0;
    while flag0==0
        rho1=rho1*0.995;
        rho2=rho2*0.995;
        MMA=[rho1 rho2; 1 0];
        if max(abs(eig(MMA)))<1
            flag0=1;
        end
    end
end

% if rho1+rho2>1
%     tempV=rho1+rho2;
%     rho1=rho1/(tempV+0.001);
%     rho2=rho2/(tempV+0.001);
% end

x=zeros(N*its,1);
eps=.0001*randn(N*its,1);
x(1)=0;
x(2)=0;
for j=3:N*its
    x(j)=rho1*x(j-1)+rho2*x(j-2)+eps(j);
end

y=zeros(1,its);
for j=1:its
    n=0;    y(j)=0;
    while n<N
        y(j)=y(j)+1/N*x(j*N-n);
        n=n+1;
    end
end

% beta=pinv(X'*X)*(X'*Y);

if rho1+rho2<1
    Y=y(3:its)'; X=[ones(its-2,1) y(2:its-1)' y(1:its-2)'];
    beta=inv(X'*X)*(X'*Y);
else
    disp('XXX')
    Y=y(4:its)'-y(3:its-1)';
    X=[ones(its-3,1) y(3:its-1)'-y(2:its-2)' y(2:its-2)'-y(1:its-3)'];
end


beta=X\Y;
rho1a=beta(2);
rho2a=beta(3);
rhoa=[rho1a rho2a];

return
