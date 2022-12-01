% R2min.m

function L = R2min(w,PD);

K=cols(PD)-1;
p=PD(:,1); P1=PD(:,2:K+1);
w(K,1)=1-sum(w);

for j=1:K
    R(j)=(P1(j,j)-P1(j,:)*w)^2;
end
L=1000*R*p;



if min(w)<0 || max(w)>1
    L=1000000000;
end
    
    