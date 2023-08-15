function u1cor = acf(amat,bmat,shockvar,yvar,dypos);
 kmax = 20; %maxc(that)+1;
 a=amat;
 omega=yvar;
 sigma=bmat*shockvar*bmat';
 % Set up indicator vectors: 
 e1= zeros(length(a),1); e2=e1; e3=e2;
 e1(dypos)=1;
 % Compute coefficients for k step ahead prediction: 
 ee=eye(length(a));
 ak=a;
 ak0=ee;
 aksum=0;
 ak0sum=0;
 by=zeros(kmax,length(a));
 dyerr=zeros(kmax,length(a));
 yyerr=by(:,1); 
 yyerrk=0; 
 i = 1;
 % Start do loop over future periods: 
 while i <= kmax;
  dyerrk=e1'*ak0;
  yyerrk=yyerrk+dyerrk*sigma*dyerrk';
  byk= e1'*(ak-ee);
  by(i,:)=byk;      % Store results for conditional coef vectors 
  dyerr(i,:)=dyerrk;
  yyerr(i)=yyerrk;  % Store results for error variances and covariances
  i = i+1;
  ak=ak*a;
  ak0=ak0*a;
 end;

 % Compute variance of growth rate variables: 
 dyvar=by(1,:)*omega*by(1,:)'+dyerr(1,:)*sigma*dyerr(1,:)';
 % Compute autocovariance at lag 1: 
 dy1cov=by(1,:)*a*omega*by(1,:)'+by(1,:)*sigma*dyerr(1,:)';
 dy2cov=by(1,:)*a*omega*by(1,:)';
 % Compute autocorrelations at lag 1: 
 dy1cor=dy1cov/dyvar;
 dy2cor=dy2cov/dyvar;
 % Stack results: 
 u1cor=[dy1cor;dy2cor];

