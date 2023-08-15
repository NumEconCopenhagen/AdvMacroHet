% R2minFit.m

function L = R2minFit(w,options)

p=options.pa;
P1=options.P;
K=cols(P);
w(K,1)=1-sum(w);

for j=1:K
    R(j)=sum((P1(:,j,j)-P1(:,:,j)*w).^2);
end
L=R*p;



if min(w)<0 || max(w)>1
    L=1000000000;
end
    
    