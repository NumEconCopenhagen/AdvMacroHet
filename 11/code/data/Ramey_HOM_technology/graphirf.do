**** GRAPHIRF.DO

***
*** Graphs impulse responses

***
*** Requires:
***     irfdat*.csv 

***************************************************************************************************

 #delimit;

drop _all;
clear all;

set more 1;

capture log close;


*********************************************;
* FIGURE 9:  Technology shocks: Several TFP shocks;
*********************************************;


import excel using techirfs.xlsx, firstrow sheet("jf_tfp");
   
   gen facjf = 0.005788/0.001814;
    foreach var in y x sp h c i { ;
     gen `var'irfb = 100*facjf*`var'irf;
	 gen `var'lob = 100*facjf*`var'lo;
	 gen `var'upb = 100*facjf*`var'up;
    };
	
  sort horiz;
  
  keep horiz yirfb xirfb spirfb hirfb cirfb iirfb ;
  save junkb.dta, replace;
  
  drop _all;
  clear all;

   import excel using techirfs.xlsx, firstrow sheet("jpt_tfp");
   
   *gen facmn = 0.005788/0.008665;  /*mn*/
    /* 0.005788/0.00554 jpt */
   gen fac = 0.005788/0.00554;
   
    foreach var in y x sp h c i { ;
     gen `var'irfc = 100*fac*`var'irf;
	 gen `var'loc = 100*fac*`var'lo;
	 gen `var'upc = 100*fac*`var'up;
    };
  sort horiz;
  
  keep horiz yirfc xirfc spirfc hirfc cirfc iirfc ;
  save junkc.dta, replace;

foreach model in ford_tfp {;

   drop _all;
   clear all;
   import excel using techirfs.xlsx, firstrow sheet("`model'");
   
     foreach var in y x sp h c i { ;
     replace `var'irf = 100*`var'irf;
	 replace `var'lo = 100*`var'lo;
	 replace `var'up = 100*`var'up;
};


merge 1:1 horiz using junkb.dta;
drop _merge; sort horiz;
merge 1:1 horiz using junkc.dta;

gen h = horiz;

tw (rarea yup ylo h, bcolor(gs14) clw(medthin medthin)) 
  (scatter yirf yirfb yirfc h, c(l l l) clp(l - .) ms(o i i) clc(blue red green) mc(blue red green) clw(medthick)) if h<=48 ,
   title("Real GDP", size(medsmall)) legend(off) xtitle("") saving(`model'_y.gph,replace);
   
 tw (rarea xup xlo h, bcolor(gs14) clw(medthin medthin)) 
  (scatter xirf  xirfb xirfc h, c(l l l) clp(l - .) ms(o i i) clc(blue red green) mc(blue red green) clw(medthick)) if h<=48 ,
   title("Labor Productivity", size(medsmall)) legend(off) xtitle("") saving(`model'_x.gph,replace);
   
 tw (rarea hup hlo h, bcolor(gs14) clw(medthin medthin)) 
 (scatter hirf hirfb hirfc h, c(l l l) clp(l - .) ms(o i i) clc(blue red green) mc(blue red green) clw(medthick)) if h<=48 ,
   title("Hours", size(medsmall)) legend(off) xtitle("") saving(`model'_h.gph,replace);
   
 tw (rarea spup splo h, bcolor(gs14) clw(medthin medthin)) 
 (scatter spirf spirfb spirfc h, c(l l l) clp(l - .) ms(o i i) clc(blue red green) mc(blue red green) clw(medthick)) if h<=48 ,
   title("Stock Prices", size(medsmall)) legend(off) xtitle("") saving(`model'_sp.gph,replace);
   
 tw (rarea cup clo h, bcolor(gs14) clw(medthin medthin)) 
  (scatter cirf cirfb cirfc h, c(l l l) clp(l - .) ms(o i i) clc(blue red green) mc(blue red green) clw(medthick)) if h<=48 ,
   title("Consumption", size(medsmall)) legend(off) xtitle("") saving(`model'_c.gph,replace);
   
 tw (rarea iup ilo h, bcolor(gs14) clw(medthin medthin)) 
  (scatter iirf iirfb iirfc h, c(l l l) clp(l - .) ms(o i i) clc(blue red green) mc(blue red green) clw(medthick)) if h<=48 ,
   title("Nonresidential Investment", size(medsmall)) legend(off) xtitle("") saving(`model'_i.gph,replace);
   
graph combine `model'_y.gph `model'_x.gph `model'_h.gph `model'_sp.gph `model'_c.gph `model'_i.gph, col(2)
  ysize(7) xsize(6) iscale(0.8);

};


drop _all;
clear all;

*********************************************;
* FIGURE 10 and 11:  Technology shocks;
*********************************************;


foreach model in bzk_ist_news jpt_mei {;

   drop _all;
   clear all;
   import excel using techirfs.xlsx, firstrow sheet("`model'");
   
     foreach var in y x sp h c i { ;
     replace `var'irf = 100*`var'irf;
	 replace `var'lo = 100*`var'lo;
	 replace `var'up = 100*`var'up;
};


*merge 1:1 horiz using junk.dta;

gen h = horiz;

tw (rarea yup ylo h, bcolor(gs14) clw(medthin medthin)) 
  (scatter yirf h, c(l l) clp(l -) ms(i i ) clc(black blue) mc(black blue) clw(medthick)) if h<=48 ,
   title("Real GDP", size(medsmall)) legend(off) xtitle("") saving(`model'_y.gph,replace);
   
 tw (rarea xup xlo h, bcolor(gs14) clw(medthin medthin)) 
  (scatter xirf  h, c(l l) clp(l -) ms(i i ) clc(black blue) mc(black blue) clw(medthick)) if h<=48 ,
   title("Labor Productivity", size(medsmall)) legend(off) xtitle("") saving(`model'_x.gph,replace);
   
 tw (rarea hup hlo h, bcolor(gs14) clw(medthin medthin)) 
 (scatter hirf h, c(l l) clp(l -) ms(i i ) clc(black blue) mc(black blue) clw(medthick)) if h<=48 ,
   title("Hours", size(medsmall)) legend(off) xtitle("") saving(`model'_h.gph,replace);
   
 tw (rarea spup splo h, bcolor(gs14) clw(medthin medthin)) 
 (scatter spirf h, c(l l) clp(l -) ms(i i ) clc(black blue) mc(black blue) clw(medthick)) if h<=48 ,
   title("Stock Prices", size(medsmall)) legend(off) xtitle("") saving(`model'_sp.gph,replace);
   
 tw (rarea cup clo h, bcolor(gs14) clw(medthin medthin)) 
  (scatter cirf  h, c(l l) clp(l -) ms(i i ) clc(black blue) mc(black blue) clw(medthick)) if h<=48 ,
   title("Consumption", size(medsmall)) legend(off) xtitle("") saving(`model'_c.gph,replace);
   
 tw (rarea iup ilo h, bcolor(gs14) clw(medthin medthin)) 
  (scatter iirf  h, c(l l) clp(l -) ms(i i ) clc(black blue) mc(black blue) clw(medthick)) if h<=48 ,
   title("Nonresidential Investment", size(medsmall)) legend(off) xtitle("") saving(`model'_i.gph,replace);
   
graph combine `model'_y.gph `model'_x.gph `model'_h.gph `model'_sp.gph `model'_c.gph `model'_i.gph, col(2)
  ysize(7) xsize(6) iscale(0.8);

};


