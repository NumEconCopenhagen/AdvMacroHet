function [param_,np,modname,neq,nlag,nlead,eqname_,eqtype_,endog_,delay_,vtype_] = ...
     compute_aim_data()

% compute_aim_data()
%     This function will return various information about the AIM model,
%     but will not compute the G and H matrices.

  eqname = cell(27, 1);
  param = cell(23, 1);
  endog = cell(27, 1);
  delay = zeros(27, 1);
  vtype = zeros(27, 1);
  eqtype = zeros(27, 1);

  modname = 'kimneoplan';
  neq = 27;
  np = 23;
  nlag = 1;
  nlead = 1;

  eqname(1) = cellstr('eq1');
  eqname(2) = cellstr('eq2');
  eqname(3) = cellstr('eq3');
  eqname(4) = cellstr('eq4');
  eqname(5) = cellstr('eq5');
  eqname(6) = cellstr('eq6');
  eqname(7) = cellstr('eq7');
  eqname(8) = cellstr('eq8');
  eqname(9) = cellstr('eq9');
  eqname(10) = cellstr('eq10');
  eqname(11) = cellstr('eq11');
  eqname(12) = cellstr('eq12');
  eqname(13) = cellstr('eq13');
  eqname(14) = cellstr('eq14');
  eqname(15) = cellstr('eq15');
  eqname(16) = cellstr('eq16');
  eqname(17) = cellstr('eq17');
  eqname(18) = cellstr('eq19');
  eqname(19) = cellstr('eq20');
  eqname(20) = cellstr('eq21');
  eqname(21) = cellstr('eq19');
  eqname(22) = cellstr('eq20');
  eqname(23) = cellstr('eq21');
  eqname(24) = cellstr('eq22');
  eqname(25) = cellstr('eq23');
  eqname(26) = cellstr('eq24');
  eqname(27) = cellstr('eq25');
  eqname_ = char(eqname);

  eqtype(1) = 1;     eqtype(2) = 1;     eqtype(3) = 1;   
  eqtype(4) = 1;     eqtype(5) = 1;     eqtype(6) = 1;   
  eqtype(7) = 1;     eqtype(8) = 1;     eqtype(9) = 1;   
  eqtype(10) = 1;     eqtype(11) = 1;     eqtype(12) = 1;   
  eqtype(13) = 1;     eqtype(14) = 1;     eqtype(15) = 1;   
  eqtype(16) = 1;     eqtype(17) = 1;     eqtype(18) = 1;   
  eqtype(19) = 1;     eqtype(20) = 1;     eqtype(21) = 1;   
  eqtype(22) = 1;     eqtype(23) = 1;     eqtype(24) = 1;   
  eqtype(25) = 0;     eqtype(26) = 0;     eqtype(27) = 0;   

  eqtype_ = eqtype;

  param(1) = cellstr('rho');
  param(2) = cellstr('delta');
  param(3) = cellstr('gama');
  param(4) = cellstr('zeta');
  param(5) = cellstr('corrtech');
  param(6) = cellstr('eta');
  param(7) = cellstr('s');
  param(8) = cellstr('tau');
  param(9) = cellstr('rscale');
  param(10) = cellstr('theta');
  param(11) = cellstr('lmy');
  param(12) = cellstr('lmx');
  param(13) = cellstr('alfa');
  param(14) = cellstr('gimmel');
  param(15) = cellstr('aleph');
  param(16) = cellstr('elast');
  param(17) = cellstr('epsilon');
  param(18) = cellstr('psim');
  param(19) = cellstr('shareI');
  param(20) = cellstr('shareC');
  param(21) = cellstr('shareG');
  param(22) = cellstr('corrgov50');
  param(23) = cellstr('corrgov3');
  param_ = char(param);

  endog(1) = cellstr('y');
  endog(2) = cellstr('n');
  endog(3) = cellstr('i');
  endog(4) = cellstr('c');
  endog(5) = cellstr('R');
  endog(6) = cellstr('r');
  endog(7) = cellstr('w');
  endog(8) = cellstr('pi');
  endog(9) = cellstr('q');
  endog(10) = cellstr('S');
  endog(11) = cellstr('v');
  endog(12) = cellstr('k');
  endog(13) = cellstr('rn');
  endog(14) = cellstr('lam');
  endog(15) = cellstr('p');
  endog(16) = cellstr('pr');
  endog(17) = cellstr('pd');
  endog(18) = cellstr('m');
  endog(19) = cellstr('x');
  endog(20) = cellstr('g');
  endog(21) = cellstr('g3');
  endog(22) = cellstr('g50');
  endog(23) = cellstr('a');
  endog(24) = cellstr('one');
  endog(25) = cellstr('enr');
  endog(26) = cellstr('et');
  endog(27) = cellstr('eg');
  endog_ = char(endog);

  delay(1) = 0;     delay(2) = 0;     delay(3) = 0;   
  delay(4) = 0;     delay(5) = 0;     delay(6) = 0;   
  delay(7) = 0;     delay(8) = 0;     delay(9) = 0;   
  delay(10) = 0;     delay(11) = 0;     delay(12) = 0;   
  delay(13) = 0;     delay(14) = 0;     delay(15) = 0;   
  delay(16) = 0;     delay(17) = 0;     delay(18) = 0;   
  delay(19) = 0;     delay(20) = 0;     delay(21) = 0;   
  delay(22) = 0;     delay(23) = 0;     delay(24) = 0;   
  delay(25) = 0;     delay(26) = 0;     delay(27) = 0;   

  delay_ = delay;

  vtype(1) = 1;     vtype(2) = 1;     vtype(3) = 1;   
  vtype(4) = 1;     vtype(5) = 1;     vtype(6) = 1;   
  vtype(7) = 1;     vtype(8) = 1;     vtype(9) = 1;   
  vtype(10) = 1;     vtype(11) = 1;     vtype(12) = 1;   
  vtype(13) = 1;     vtype(14) = 1;     vtype(15) = 1;   
  vtype(16) = 1;     vtype(17) = 1;     vtype(18) = 1;   
  vtype(19) = 1;     vtype(20) = 1;     vtype(21) = 1;   
  vtype(22) = 1;     vtype(23) = 1;     vtype(24) = 2;   
  vtype(25) = 1;     vtype(26) = 1;     vtype(27) = 1;   

  vtype_ = vtype;



