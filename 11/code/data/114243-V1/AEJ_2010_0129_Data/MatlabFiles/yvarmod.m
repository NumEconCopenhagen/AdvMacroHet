function ysigma = yvarmod(amat,bmat,shockvar);
  yindex=find((sum(abs(amat))==0)'); %Find zero columns of companion form
  zindex=find((sum(abs(amat))>0)');  %Find nonzero columns
  az=amat(zindex,zindex);
  k=length(az(1,:));
  ay=amat(:,zindex);
  bz=bmat(zindex,:);
  by=bmat(yindex,:);
  ak=eye(length(az(:,1)));
  bsigma=bz*shockvar*bz';
  zsigma=bsigma;
  i=1;
  while i<200;
   ak=ak*az;
   zsigma=zsigma+ak*bsigma*ak';
   i=i+1;
  end;
  ysigma=ay*zsigma*ay'+bmat*shockvar*bmat';
%  ysigma(9,1)
%  ysigma(10,1)



function y1 = find(x);
% return index for rows where x==1 
  y = (1:length(x))';            %seqa(1,1,rows(x));
  y = y.*(x==1);
  y = find(y);
  if max(x)<=0; 
   y = 0; 
  end;
  y1=y;


