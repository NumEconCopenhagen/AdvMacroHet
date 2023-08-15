function yvec1=simffix(a,b,nstep,simshock,begsh,endsh);
%*******************************************************
%******** Proc to simulate system:                      
% simffix takes a ROW vector called simshock made up of 
% simulated draws from some distribution and applies them
% to the amat system to simulate the model.                        
%*******************************************************

% e=shockvar*randn(length(b(1,:)),1);            % initial shock 
 e=zeros(length(b(1,:)),1);                 % @ initial shock @
 e(begsh:endsh) = simshock(:,1);
 y=b*e;                                 % starting value 
 yvec=zeros(nstep,length(a));             % matrix to store results 
 yvec(1,:)=y';                          % store initial value 
 i=2;
 while i<=nstep;                     % loop through steps 
  %e=shockvar*randn(length(b(1,:)),1);           % new shock vector 
  e(begsh:endsh) = simshock(:,i);
  y=a*y+b*e;                            % update y 
  yvec(i,:)=y';                         % store result 
  i=i+1;
 end;                                  % end loop 

yvec1 = yvec;

