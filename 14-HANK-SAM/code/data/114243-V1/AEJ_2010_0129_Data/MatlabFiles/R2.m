% R2: takes 2 series and returns R2, no regression (i.e. assume constant =0
% and coefficient of 1)

function R = R2(y,x)

SStot=sum((y-mean(y)).^2);
SSerr=sum((y-x).^2);

R=1-SSerr/SStot;

return