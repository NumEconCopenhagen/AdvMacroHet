

% rc adds in effort stuff here july 14, 1999
%********************************************************************
%*** Written by Simon Gilchrist. Dec 15  1996. **********************
%*** Last modified July 1997                                         
%********************************************************************
%********************************************************************
%     Program to solve and simulate log-linear RBC model             
%********************************************************************
clear all;

% 7/13/99  calculate various measures of Solow Residuals
svec = [0.65 1 1.9 2.2];


%#include e:\willis\rclbd\aimprc.prg;   % Call AIM subroutines (procedures) 
%library pgraph;             % Load graphics files 
%
plotflag=1; % Set to one if you want to see plots 
%plotflag=1; % Set to one if you want to see plots 
%               % Hit escape to get out of a plot 
%
% Model Parameters                                                       
  alpha = 0.35;     % Cobb-Douglas Share of Capital                      
  omega = 0.65;     % Cobb-Douglas Share of Labor                        
  beta = 0.99;      % Discount factor                                    
  kdelta = 0.025;   % Capital dep.                                       
  arho = 0.95;      % Persistence of aggregate technology shock 0<=rho<=1
  asigma = 0.01;    % Std.err of tech. shocks 

  grho = 0.95;      % Persistence of govt. spending shock 
  gsigma = 0;       % Std. err of govt. spending shocks 

  v = 3.65;         % Coef on utility: u=ln(c)+v(1-L)                    
                    % (Hansen/Rogerson preferences)                      

% Compute nonstochastic steady state:                                    

  a=1;                               % Steady state value of technology  
  q = 1;                             % Price of capital goods            
%  q = 5;                             % Price of capital goods            
  rr = q/beta;                       % Rate of return: q(1-delta)+mpk    
  yk = (1/alpha)*(rr-q*(1-kdelta));  % output/capital ratio              
  ik = kdelta;                       % invest/capital ratio              
  ck = yk-ik;                        % consumption of workers            
  h = (omega/v)*(yk/ck);             % labor hours                       
  k = (a*(h^omega)/yk)^(1/(1-alpha));% capital                           

  c=ck*k;            % Steady state level of consumption                 
  y=yk*k;            % Steady state level of income                      
  i=ik*k;            % Steady state level of investment                  

  % comment: model allows for govt. spending that is zero on average and 
  % therefore does not affect steady state 

%******************************************************************
%*************** End steady state computations ********************
%******************************************************************

%******************************************************************
%**** Begin setup for constructing log-linear system coef matrix **
%******************************************************************

  nlead = 1;  % Number of leads in system 
  nlag = 1;   % Number of lags in system 

% Coefficient indicators: 
% (note xnames is not used for anything right now except to 
% determine number of equations "neq" 

  xnames = {'c' 'y' 'h' 'i' 'k' 'r' 'dc' 'dy' 'di' 'dk' 'dh' 'apl' 'st' 's1' 's2' 's3' 's4' 'a' 'g' 'aa' 'ag'};
  xnum = length(xnames); % Number of variables in system 
  neq = xnum;
necuti=neq
  % Ordering of contemporaneous variables: 
% These will be used to construct position indicators 
% for leads and lags when constructing coefficient matrix 
%                                                         
% Note: All shock variables must be placed last in the system for 
% impulse response and simulation programs to work properly 

  cpos  = 1;        % consumption 
  ypos  = 2;        % output 
  hpos  = 3;        % hours 
  ipos  = 4;        % investment 
  kpos  = 5;        % capital 
  rpos  = 6;        % interest rate 
  dcpos = 7;        % consumption growth
  dypos = 8;        % output growth 
  dipos = 9;        % investment growth
  dkpos = 10;        % capital growth 
  dhpos = 11;        % hours growth 
  aplpos= 12;        % average product of labor
  stpos = 13;        % Solow residual
  s1pos = 14;        % Solow residual
  s2pos = 15;        % Solow residual
  s3pos = 16;        % Solow residual
  s4pos = 17;        % Solow residual
  apos  = 18;        % technology series 
  gpos  = 19;        % govt. spending series 
  aapos = 20;        % technology shock (placed last)
  agpos = 21;        % govt. spending shock 

  shockvec=[aapos;agpos]; % Identify position of shock variables 
  shockvar=diag([asigma^2 gsigma^2]); % Var/Covar matrix for shocks 

% Indicators for contemporanous coefs for each variable: 

  colzero = 0+nlag*xnum;  % Position counter for start of contemp. coefs 
  collead = 0+nlag*xnum+xnum; % Position counter for start of lead coefs 
  collag = 0;                 % Position counter for start of lag coefs  

  czero = colzero+cpos;
  yzero = colzero+ypos;
  hzero = colzero+hpos;
  izero = colzero+ipos;
  kzero = colzero+kpos;
  rzero = colzero+rpos;
  dczero = colzero+dcpos;
  dyzero = colzero+dypos;
  dizero = colzero+dipos;
  dkzero = colzero+dkpos;
  dhzero = colzero+dhpos;
  aplzero = colzero+aplpos;
  stzero = colzero+stpos;
  s1zero = colzero+s1pos;
  s2zero = colzero+s2pos;
  s3zero = colzero+s3pos;
  s4zero = colzero+s4pos;
  azero = colzero+apos;
  gzero = colzero+gpos;
  aazero = colzero + aapos;
  agzero = colzero + agpos;


% Indicators for lead coefficients for each variable: 

  clead = collead+cpos;
  ylead = collead+ypos;
  hlead = collead+hpos;
  ilead = collead+ipos;
  klead = collead+kpos;
  rlead = collead+rpos;
  dylead = collead+dypos;
  alead = collead+apos;
  glead = collead+gpos;
  aalead = collead+aapos;
  aglead = collead+agpos;

% Indicators for lag coefficients for each variable: 

  clag = collag+cpos;
  ylag = collag+ypos;
  hlag = collag+hpos;
  ilag = collag+ipos;
  klag = collag+kpos;
  rlag = collag+rpos;
  dylag = collag+dypos;
  alag = collag+apos;
  glag = collag+gpos;
  aalag = collag+aapos;
  aglag = collag+agpos;

% Determine number of coefficients per equation: 
neq
nequ=neq

  ncoef = neq*(nlag+nlag+1);
  cof = zeros(neq,ncoef);             % Coef matrix 
                                      % Each row is an equation 
% Setup coefficients vectors for each equation: 
% ==============================================

% c = y-h 
  cof1 = zeros(ncoef,1);
  cof1(czero)=1;
  cof1(yzero)=-1;
  cof1(hzero)=1;

% 1 = c(t)-c(t+1)+R(t+1) 
  cof2 = zeros(ncoef,1);
  cof2(czero)=1;
  cof2(clead)=-1;
  cof2(rlead)=1;

% R(t) =  ((alpha*y/k)/rr)*(y(t)-k(t))           
  cof3 = zeros(ncoef,1);
  rr = (1-kdelta)*q + alpha*y/k;
  cof3(rzero) = 1;
  cof3(yzero) = -(alpha*y/k)/rr;
  cof3(klag) = alpha*(y/k)/rr;

% k(t)=(1-kdelta)*k(t-1)+kdelta*i(t-1) 
  cof4 = zeros(ncoef,1);
  cof4(kzero)=1;
  cof4(klag)=-(1-kdelta);
  cof4(izero)=-kdelta;

% y(t) = (c/y)*c(t)+(i/y)*i(t) + g(t) 
  cof5 = zeros(ncoef,1);
  cof5(yzero)=1;
  cof5(czero)=-c/y;
  cof5(izero)=-i/y;
  cof5(gzero)=-1;

% y(t) = alpha*k(t-1) + omega*h(t) + a(t) 
  cof6 = zeros(ncoef,1);
  cof6(yzero)=1;
  cof6(klag)=-alpha;
  cof6(hzero)=-omega;
  cof6(azero)=-1;

% 0= dc(t)- c(t) + c(t-1) 
  cof7 = zeros(ncoef,1);
  cof7(dczero)= 1;
  cof7(czero) =-1;
  cof7(clag)  = 1;

% 0= dy(t)- y(t) + y(t-1) 
  cof8 = zeros(ncoef,1);
  cof8(dyzero)= 1;
  cof8(yzero) =-1;
  cof8(ylag)  = 1;

% 0= di(t)- i(t) + i(t-1) 
  cof9 = zeros(ncoef,1);
  cof9(dizero)= 1;
  cof9(izero) =-1;
  cof9(ilag)  = 1;

% 0= dk(t)- k(t) + k(t-1) 
  cof10 = zeros(ncoef,1);
  cof10(dkzero)= 1;
  cof10(kzero) =-1;
  cof10(klag)  = 1;

% 0= dh(t)- h(t) + h(t-1) 
  cof11 = zeros(ncoef,1);
  cof11(dhzero)= 1;
  cof11(hzero) =-1;
  cof11(hlag)  = 1;

% 0=apl(t) - y(t) + h(t);
  cof12 = zeros(ncoef,1);
  cof12(aplzero)= 1;
  cof12(yzero)= -1;
  cof12(hzero)= 1;

% 0 = st(t) - y(t) + 0.65*h(t) + 0.35*k(t)
  cof13 = zeros(ncoef,1);
  cof13(stzero)= 1;
  cof13(yzero)= -1;
  cof13(hzero)= 0.65;
  cof13(kzero)= 0.35;

% 0 = s1(t) - y(t) + 0.65*h(t) + 0.35*k(t)
  cof14 = zeros(ncoef,1);
  cof14(s1zero)= 1;
  cof14(yzero)= -1;
  cof14(hzero)= svec(1);
  cof14(kzero)= 0.35;
% 0 = s2(t) - y(t) + 0.65*h(t) + 0.35*k(t)
  cof15 = zeros(ncoef,1);
  cof15(s2zero)= 1;
  cof15(yzero)= -1;
  cof15(hzero)= svec(2);
  cof15(kzero)= 0.35;
% 0 = s3(t) - y(t) + 0.65*h(t) + 0.35*k(t)
  cof16 = zeros(ncoef,1);
  cof16(s3zero)= 1;
  cof16(yzero)= -1;
  cof16(hzero)= svec(3);
  cof16(kzero)= 0.35;
% 0 = s4(t) - y(t) + 0.65*h(t) + 0.35*k(t)
  cof17 = zeros(ncoef,1);
  cof17(s4zero)= 1;
  cof17(yzero)= -1;
  cof17(hzero)= svec(4);
  cof17(kzero)= 0.35;

% a(t) = arho*a(t-1) + aa(t) 
  cof18 = zeros(ncoef,1);
  cof18(azero)=1;
  cof18(alag)=-arho;
  cof18(aazero)=-1;

% g(t) = grho*g(t-1) + ag(t) 
  cof19 = zeros(ncoef,1);
  cof19(gzero)=1;
  cof19(glag)=-grho;
  cof19(agzero)=1;

% Technology Shock: 
  cof20 = zeros(ncoef,1);
  cof20(aazero)=1;

% Govt. spending shock 
  cof21 = zeros(ncoef,1);
  cof21(agzero)=1;


% Concatenate coef vectors across equations: 

nequii=neq
  cof=[cof1';cof2';cof3';cof4';cof5';cof6';cof7';cof8';cof9';cof10';cof11';cof12';cof13';cof14';cof15';cof16';cof17';cof18';cof19';cof20';cof21'];

%*****************************************************************
%************ End coefficient matrix setup ***********************
%*****************************************************************

%*****************************************************************
%*************Begin Solution Algorithm ***************************
%*****************************************************************
%                                                                 
%  Solve a linear perfect foresight model using the gauss eig     
%  function to find the invariant subspace associated with the big 
%  roots.  This procedure will fail if the companion matrix is     
%  defective and does not have a linearly independent set of       
%  eigenvectors associated with the big roots.                     
%                                                                  
%  Input arguments:                                                
%                                                                  
%    h         Structural coef matrix (neq,neq*(nlag+1+nlag)).     
%    neq       Number of equations.                                
%    nlag      Number of lags.                                     
%    nlag     Number of lags.                                      
%    condn     lag tolerance used as a condition number test       
%              by numeric_shift and reduced_form.                  
%    uprbnd    Inclusive upper bound for the modulus of roots      
%              allowed in the reduced form.                        
%                                                                  
%  Output arguments:                                               
%                                                                  
%    cofb      Reduced form coefficient matrix (neq,neq*nlag).     
%    scof      Observable Structure                                
%    amat      Companion form matrix                               
%    b         Contemporaneous coefficient matrix                  
%    Model satisfies:                                              
%                                                                  
%    z(t) = amat*z(t-1) + b*e(t)                                   
%                                                                  
%    where the first neq elements of z(t) are the contemporaneous  
%    values of the variables in the model                          
%    and e(t) is the shock vector of conformable dimension         
%                                                                  
%    rts       Roots returned by eig.                              
%    ia        Dimension of companion matrix (number of non-trivial
%              elements in rts).                                   
%    nexact    Number of exact shiftrights.                        
%    nnumeric  Number of numeric shiftrights.                      
%    lgroots   Number of roots greater in modulus than uprbnd.     
%    mcode     Return code: see function aimerr.                   
%*****************************************************************

%  #include aimprc0.prg; %  % Call AIM subroutines (procedures) 

% Use AIM procedure to solve model: 

  uprbnd = 1+1e-8;    % Tolerance values for AIM program 
  condn = 1e-8;


% ---------------------------------------------------------------------
% Run AIM
% ---------------------------------------------------------------------
neq
[cofb,rts,ia,nex,nnum,lgrts,mcode] = ...
       aim_eig(cof,neq,nlag,nlead,condn,uprbnd);



%disp(['Number of exact shiftrights (nex):    ',num2str(nex)]);
%disp(['Number of numeric shiftrights (nnum): ',num2str(nnum)]);
%disp(['Number of large roots (lgrts):        ',num2str(lgrts)]);
%disp(['Number of stability conditions (nex + nnum + lgrts) -'])
%disp(['       number required (neq*nlead) = ',num2str(nex+nnum+lgrts-neq*nlead)]);
%disp(['Dimension of state transition matrix (ia): ',num2str(ia)]);
%errstr = aimerr(mcode);
%disp(errstr);


% ---------------------------------------------------------------------
% Display roots, magnitude of roots and period
% ---------------------------------------------------------------------

%[amp,per] = vibz(rts,0);

% ---------------------------------------------------------------------
% Check accuracy of solution
% ---------------------------------------------------------------------

%[q,err] = checkaim(neq,nlag,nlead,cof,cofb);

% ---------------------------------------------------------------------
% Compute observable structure
% ---------------------------------------------------------------------

scof = obstruct(cof,cofb,neq,nlag,nlead);
%scof1 = obstr_t1(cof,cofb,neq,nlag,nlead);


% need to calculate amat and b
% ===============================
s0 = scof(:,(neq*nlag+1):neq*(nlag+1)); %Contemp. coefs from obs. structure
amat=zeros(neq*nlag,neq*nlag);   % Initialize A matrix 
bmat=cofb(1:neq,((nlag-1)*neq+1):nlag*neq);  % Lag 1 coefficients 
i1=2;
while i1<=nlag;
  bmat=[bmat cofb(1:neq,((nlag-i1)*neq+1):(nlag-i1+1)*neq)]; % Lag i coefs 
  i1=i1+1;
end;
amat(1:neq,:)=bmat;  % Coefs for equations 
if nlag>1;
 amat((length(cofb(:,1))+1):length(amat(:,1)),1:neq*(nlag-1))=eye(neq*(nlag-1));
end;
b = zeros(length(amat(:,1)),length(s0(1,:)));
b(1:length(s0(:,1)),1:length(s0(1,:))) = inv(s0);  % Store coefs 
b=b(:,shockvec);

% COMPUTE THE COVARIANCE MATRIX FOR THE MODEL -- THEN GET CORRELATIONS 
%shockvar = zeros(xnum,xnum);
%shockvar[apos,apos] = asigma^2;
omega = yvarmod(amat,b,shockvar);

domega = diag(omega)+1e-30;
sdo = domega.^0.5;
denom = sdo*sdo';
omcorr = omega./denom;
sdratio = sdo/sdo(ypos);

%output file = d:\willis\rclbd\stats.txt on;%   % save statistics 
%print tech;
disp(['output standard deviation (%)       = ' num2str(sdo(ypos))]);
disp(['consumption standard deviation (%)  = ' num2str(sdo(cpos))]);
disp(['hours standard deviation (%)        = ' num2str(sdo(hpos))]);
disp(['investment standard deviation (%)   = ' num2str(sdo(ipos))]);
disp(['consumption/output std dev ratio    = ' num2str(sdratio(cpos))]);
disp(['hours/output std dev ratio          = ' num2str(sdratio(hpos))]);
disp(['investment/output std dev ratio     = ' num2str(sdratio(ipos))]);

% report some of the correlations from omcorr
%disp('correlations of (c,y,h, i,k) levels with y')
%omcorr(2,1:5)

%disp('correlations of (c,y,h, i,k) growth rates with y')
%omcorr(8,7:11)


% CALCULATE MOMENTS FOR THE VARIOUS MEASURES OF THE SOLOW RESIDUALS
sscorr = omcorr(stpos,s1pos:s4pos);
sycorr = omcorr(ypos,s1pos:s4pos);
shcorr = omcorr(hpos,s1pos:s4pos);
sicorr=omcorr(ipos,s1pos:s4pos);

disp(' ')
disp(['S^ values          =  ' num2str(svec)]);
disp(['Correlation with S =  ' num2str(sscorr)]);
disp(['Correlation with Y =  ' num2str(sycorr)]);
disp(['Correlation with H =  ' num2str(shcorr)]);
disp(['Correlation with I =  ' num2str(sicorr)]);


%GET ACF FUNCTION
%u1cor = acf(amat,b,shockvar,omega,dypos);
%u1cor = acfjw(amat,b,shockvar,omega,dypos,ypos)
%acfdy = u1cor(:,1);
%acfy  = u1cor(:,2);

% ========================================================
% Compute Impulse response function using companion form     
% solution matrix amat and contemporaneous matrix b obtained 
% from aim_run procedure: 
% ========================================================
% Technology shock: 

shock=zeros(length(shockvec(:,1)),1); % Shock vector 
shock(1,1)=1;                  % Shock variable, size of shock 
nstep=100;                      % Number of steps in impulse response 
imptech = impf(amat,b,shock,nstep,neq); 

imptechi=imptech(1:10,:)
% Call impulse respons proc 
dat=(1:nstep)';           % Date variable for plotting 


if plotflag==1;
 colordef black
 figure(1)
 colordef black
 plot(dat,imptech(1:nstep,apos),'g-',...
      dat,imptech(1:nstep,cpos),'b-',...
      dat,imptech(1:nstep,ypos),'r-',...
      dat,imptech(1:nstep,kpos),'c:',...
      dat,imptech(1:nstep,hpos),'y--')
 colordef black
 legend('tech','consumption','output','capital','labor',-1);
 title('Technology Shock Impulse Response Function')
 xlabel('Periods');
 ylabel('Percent deviation from steady state');
end;

%outwidth 250;
%output file = e:\willis\rclbd\rbctech.txt reset;   % save impulse response functions 
%screen off; 
%print "     CONSUMPTION          OUTPUT          LABOR          INVESTMENT        CAPITAL      INTEREST RATE       TECHNOLOGY     GOVERNMENT         TECH SHOCK      GOVT. SHOCK";
%print imptech[.,.]; screen on; output off;

%imptech(1:nstep,apos)


% =================================
% SIMULATION
% =================================
simstep = 1000;
yvec1=simf(amat,b,shockvar,simstep,neq);

acf = acfsim(yvec1);


% double check the correlations
sscorr = corrcoef(yvec1(:,stpos),yvec1(:,s2pos))
sycorr = corrcoef(yvec1(:,ypos),yvec1(:,s2pos))
shcorr = corrcoef(yvec1(:,hpos),yvec1(:,s2pos))

% rebuild the levels
%yy = y/(k^alpha*h^(1-alpha));
%y4 = y/(k^alpha*h^2.2);
%s4 = (yvec1(:,s4pos)+1)*y4;
%st = (yvec1(:,stpos)+1)*yy;
%corrcoef(s4,st)
%corrcoef(log(s4),log(st))

%disp(' ')
%disp('HERE ARE THE CONTEMPORANEOUS CORRELATIONS IN LEVELS AND GROWTH RATES')
%disp(['corr(y,c)   = ' num2str(omcorr(ypos,cpos)) '   ' num2str(omcorr(dypos,dcpos))]);
%disp(['corr(y,h)   = ' num2str(omcorr(ypos,hpos)) '   ' num2str(omcorr(dypos,dhpos))]);
%disp(['corr(y,i)   = ' num2str(omcorr(ypos,ipos)) '   ' num2str(omcorr(dypos,dipos))]);
%disp(['corr(y,k)   = ' num2str(omcorr(ypos,kpos)) '   ' num2str(omcorr(dypos,dkpos))]);


%disp(' ')
%disp('FROM SIMULATIONS -- CONTEMPORANEOUS CORRELATIONS IN LEVELS AND GROWTH RATES')
%yccorr = corrcoef(yvec1(:,ypos),yvec1(:,cpos));
%yhcorr = corrcoef(yvec1(:,ypos),yvec1(:,hpos));
%yicorr = corrcoef(yvec1(:,ypos),yvec1(:,ipos));
%ykcorr = corrcoef(yvec1(:,ypos),yvec1(:,kpos));
%dyccorr = corrcoef(yvec1(:,dypos),yvec1(:,dcpos));
%dyhcorr = corrcoef(yvec1(:,dypos),yvec1(:,dhpos));
%dyicorr = corrcoef(yvec1(:,dypos),yvec1(:,dipos));
%dykcorr = corrcoef(yvec1(:,dypos),yvec1(:,dkpos));
%disp(['corr(y,c)   = ' num2str(yccorr(1,2)) '   ' num2str(dyccorr(1,2))]);
%disp(['corr(y,h)   = ' num2str(yhcorr(1,2)) '   ' num2str(dyhcorr(1,2))]);
%disp(['corr(y,i)   = ' num2str(yicorr(1,2)) '   ' num2str(dyicorr(1,2))]);
%disp(['corr(y,k)   = ' num2str(ykcorr(1,2)) '   ' num2str(dykcorr(1,2))]);



