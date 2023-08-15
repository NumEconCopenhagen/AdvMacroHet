function d = duplication(n)
% duplication(n)
% Returns Magnus and Neudecker's duplication matrix of size n
if 0
  % first method
  a = zeros(n);
  k = 1;
  for j = 1:n
    for i = 1:n
      if i >= j
	a(i,j) = k;
	k = k + 1;
      else
	a(i,j) = a(j,i);
      end
    end
  end
else
  % second method
  a = tril(ones(n));
  i = find(a);
  a(i) = 1:length(i);
  a = a + tril(a,-1)';
end
j = vec(a);

m = n*(n+1)/2;
d = zeros(n*n,m);
for r = 1:rows(d)
  d(r, j(r)) = 1;
end
