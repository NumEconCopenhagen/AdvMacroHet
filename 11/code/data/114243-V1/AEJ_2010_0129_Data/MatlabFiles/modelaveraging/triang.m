%%% This file yields the triangular factorization of a positive definite
%%% matrix Omega.  This is based on Hamilton, pages 87-90.

function [A, D] = triang(Omega)

[K,T]=size(Omega);
n=K;

H=Omega;

for j=1:n-1
E(:,:,j)=eye(n,n);
end

for j=1:n-1
    for k=j+1:n
        E(k,j,j)=-H(k,j)/H(j,j);
    end
    H=E(:,:,j)*H*E(:,:,j)';
end

A=inv(E(:,:,1));
for j=2:n-1
    A=A*inv(E(:,:,j));
end

D=inv(A)*Omega*inv(A');

return
