**** FEVDVAR.DO

***  Valerie A. Ramey "Macroeconomic Shocks and Their Propagation" Handbook of Macroeconomics

***  Technology shock results


***
*** Requires:
***     Technology_data.xlsx 

***************************************************************************************************

 #delimit;

drop _all;
clear all;

set more 1;
set matsize 1000;
set mem 400m;

capture log close;
log using fevdvartech_results.log, replace;

/*******************************************************************************
** RAW DATA IMPORTATION AND DATA SETUP

** SEE README SHEET OF EXCEL FILE FOR VARIABLE DEFINTIIONS
*******************************************************************************/

import excel Technology_data, firstrow sheet("techdat") case(lower);


gen qdate = q(1947q1) + _n-1;
tsset qdate, q;

/*******************************************************************************
** CONSTRUCT VARIABLES
*******************************************************************************/

* DEFINE QUADRATIC TREND;

gen t = _n;
gen t2 = t^2;

gen xtot = rgdp/tothours;
gen rstockp = stockp_sh/pgdp;

foreach var in rgdp rcons rnri ybus tothours rstockp nbus {;
  replace `var' = 1000*`var'/pop;
};

foreach var in rgdp rcons rnri xtot ybus xbus nbus tothours pgdp rstockp {;
  
  gen l`var' = ln(`var');
  gen dl`var' = D.l`var';

  quietly gen bl`var' = .;
  quietly gen up90l`var' = .;
  quietly gen lo90l`var' = .;
  
}; 

foreach var in ltfp ltfp_util ltfp_i ltfp_i_util { ;

  gen d`var' = 400*D.`var';
  
};


* CREATE BIVARIATE LR RESTRICTION SHOCKS;

gen d2lnbus = D.dlnbus;
gen d2ltothours = D.ltothours;


ivreg2 dlxbus (L(0/3).d2lnbus = L(1/4).dlnbus) L(1/4).dlxbus ;
predict gali_tfp, resid;

ivreg2 dlxbus (L(0/3).dlnbus = L(1/4).lnbus) L(1/4).dlxbus t t2;
predict cev_tfp, resid;


* JOHN FERNALD SHOCKS;

gen jf_tfp = dltfp_util;
gen jf_ist = dltfp_i_util;

gen bp_tfp_news = bp_tfp_news_sr;


*local shock mfev;


local p = 4;

/*foreach shock in gali_tfp cev_tfp jf_tfp jf_ist ford_tfp bp_tfp_news bs_tfp_news bzk_ist_news bzk_ist
  bzk_tfp jpt_tfp jpt_mei jpt_ist mn_tfp_p mn_ist_p mn_tfp_s mn_ist_s mn_tfp_p_n4 mn_tfp_p_n8
   mn_ist_p_n4 mn_ist_p_n8 mn_tfp_s_n4	mn_tfp_s_n8	mn_ist_s_n4	mn_ist_s_n8 {;*/
   
 foreach shock in ford_tfp jf_tfp jpt_tfp bzk_ist_news jpt_mei {;
   

   quietly var `shock' lrgdp ltothours lrstockp lrcons lrnri, lags(1/`p') exog(t t2);

   irf create irf, step(21) set(irf`shock', replace) nose;
   irf table fevd, impulse(`shock') response(lrgdp ltothours) noci;

};
 

capture log close;
