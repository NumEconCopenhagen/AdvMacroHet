function Tv = Tau(M,v)

% this function calculates the autocovariance matrix of vector m with lag v

T=cols(M);

Tv=zeros(rows(M));
j=v+1;
while j<T+1
    Tv=Tv+(1/T)*M(:,j)*M(:,j-v)';
    j=j+1;
end



