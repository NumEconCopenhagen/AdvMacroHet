function yvec1=simf(a,b,shockvar,nstep,neq);
%*******************************************************
%******** Proc to simulate system:                      
% Proc simulates model using error structure:           
%         e(t) = shockvar*u(t)                          
% where shockvar is an neq x neq var/covar matrix of    
% e(t) and                                              
%                                                       
% u(t) is an neq x 1 vector of random draws from a      
% standard normal distribution                          
%*******************************************************

 e=shockvar*randn(length(b(1,:)),1);            % initial shock 
 y=b*e;                                 % starting value 
 yvec=zeros(nstep,length(a));             % matrix to store results 
 yvec(1,:)=y';                          % store initial value 
 i=2;
 while i<=nstep;                     % loop through steps 
  e=shockvar*randn(length(b(1,:)),1);           % new shock vector 
  y=a*y+b*e;                            % update y 
  yvec(i,:)=y';                         % store result 
  i=i+1;
 end;                                  % end loop 

yvec1 = yvec;

