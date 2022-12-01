**** JORDATECH.DO

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
log using jordatech_results.log, replace;

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


foreach var in rgdp rcons rnri ybus tothours nbus rstockp {;
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

/*
ivreg2 dlxtot (L(0/3).d2ltothours = L(1/4).dltothours) L(1/4).dlxtot ;
predict gali_tfp, resid;

ivreg2 dlxtot (L(0/3).dltothours = L(1/4).ltothours) L(1/4).dlxtot t t2;
predict cev_tfp, resid;
*/

* JOHN FERNALD SHOCKS;

gen jf_tfp = dltfp_util;
gen jf_ist = dltfp_i_util;


* LET H BE THE HORIZON;

gen h = t - 1 ;


**************************************************************;
*  CORRELATIONS BETWEEN SHOCKS: Table 10;
**************************************************************;

*non-news tfp;

corr gali_tfp cev_tfp jf_tfp;

corr gali_tfp cev_tfp jf_tfp ford_tfp bzk_tfp jpt_tfp;
corr gali_tfp cev_tfp jf_tfp ford_tfp bzk_tfp jpt_tfp mn_tfp_p mn_tfp_s;

*non-news ist;

corr jf_ist bzk_ist;
corr jf_ist bzk_ist jpt_mei jpt_ist mn_ist_p mn_ist_s;

*news tfp;

corr bp_tfp_news_sr bp_tfp_news_lr mn_tfp_p_n4 mn_tfp_p_n8 mn_tfp_s_n4 mn_tfp_p_n8;
corr bp_tfp_news_sr bp_tfp_news_lr bs_tfp_news mn_tfp_p_n4 mn_tfp_p_n8 mn_tfp_s_n4 mn_tfp_p_n8;

*news ist;

corr bzk_ist_news mn_ist_p_n4 mn_ist_p_n8 mn_ist_s_n4 mn_ist_s_n8;


********************************************************************;
* AUTOCORRELATION AND GRANGER CAUSALITY: Table 11;
********************************************************************;

foreach var in gali_tfp cev_tfp jf_tfp jf_ist ford_tfp bp_tfp_news_sr bs_tfp_news bzk_ist_news bzk_ist
bzk_tfp jpt_tfp jpt_mei jpt_ist mn_tfp_p mn_ist_p mn_tfp_s mn_ist_s mn_tfp_p_n4 mn_tfp_p_n8
mn_ist_p_n4 mn_ist_p_n8 mn_tfp_s_n4	mn_tfp_s_n8	mn_ist_s_n4	mn_ist_s_n8 {;

  reg `var' L(1/2).`var';
  test L.`var' L2.`var';
  scalar plags_`var' = r(p);
  reg `var' L(1/2).`var' L(1/2).lrgdp L(1/2).lrstockp L(1/2).lrcons;
  test L.lrgdp L2.lrgdp L.lrstockp L2.lrstockp L.lrcons L2.lrcons;
  scalar pgc_`var' = r(p);

};

foreach var in gali_tfp cev_tfp jf_tfp jf_ist ford_tfp bp_tfp_news_sr bs_tfp_news bzk_ist_news bzk_ist
bzk_tfp jpt_tfp jpt_mei jpt_ist mn_tfp_p mn_ist_p mn_tfp_s mn_ist_s mn_tfp_p_n4 mn_tfp_p_n8
mn_ist_p_n4 mn_ist_p_n8 mn_tfp_s_n4	mn_tfp_s_n8	mn_ist_s_n4	mn_ist_s_n8 {;
   
   scalar list plags_`var';
 
 };
 
 foreach var in gali_tfp cev_tfp jf_tfp jf_ist ford_tfp bp_tfp_news_sr bs_tfp_news bzk_ist_news bzk_ist
bzk_tfp jpt_tfp jpt_mei jpt_ist mn_tfp_p mn_ist_p mn_tfp_s mn_ist_s mn_tfp_p_n4 mn_tfp_p_n8
mn_ist_p_n4 mn_ist_p_n8 mn_tfp_s_n4	mn_tfp_s_n8	mn_ist_s_n4	mn_ist_s_n8 {;
   
   scalar list pgc_`var';
 
 };



* RUN JORDA PROCEDURE AND GRAPH THE IRFS: Figure 9;

******************************************************************************;
/*Shocks: gali_tfp cev_tfp jf_tfp jf_ist ford_tfp bp_tfp_news bs_tfp_news_sr bzk_ist_news bzk_ist
bzk_tfp jpt_tfp jpt_mei jpt_ist mn_tfp_p mn_ist_p mn_tfp_s mn_ist_s mn_tfp_p_n4 mn_tfp_p_n8
mn_ist_p_n4 mn_ist_p_n8 mn_tfp_s_n4	mn_tfp_s_n8	mn_ist_s_n4	mn_ist_s_n8*/


* Choice: ford_tfp jf_tfp jpt_tfp jpt_mei bzk_ist_news bp_tfp_news_sr;

global shock jf_tfp;

local p = 2;  

forvalues i = 0/20 {;

*foreach var in lrgdp lxtot lybus lxbus ltothours lnbus lrfixinv lpgdp lrstockp {;

foreach var in lrgdp lxtot lrstockp ltothours lrcons lrnri {;

newey F`i'.`var' L(0/`p').$shock L(1/`p').lrgdp L(1/`p').lrstockp L(1/`p').lxtot L(1/`p').`var' t t2, lag(`=`i' + 1');

  gen b`var'h`i' = _b[$shock];
  
  gen se`var'h`i' = _se[$shock];

  quietly replace b`var' = b`var'h`i' if h==`i'; 
 
  quietly replace up90`var' = b`var'h`i' + 1.68*se`var'h`i' if h==`i';
  quietly replace lo90`var' = b`var'h`i' - 1.68*se`var'h`i' if h==`i';

  
};

};

* Outputted to .csv file which is copied into techirf.xlxs to produce nice looking graphs;

outsheet h blrgdp lo90lrgdp up90lrgdp blxtot lo90lxtot up90lxtot blrstockp lo90lrstockp up90lrstockp
  bltothours lo90ltothours up90ltothours blrcons lo90lrcons up90lrcons
  blrnri lo90lrnri up90lrnri using junk.csv if h<=20, replace comma;
  

foreach var in lrgdp lxtot lrstockp ltothours lrcons lrnri  { ;

tw (rarea up90`var' lo90`var' h, bcolor(gs14) clw(medthin medthin)) 
  (scatter b`var' h, c(l ) clp(l ) ms(i ) clc(black) mc(black) clw(medthick)) if h<=20,
  saving(`shock'_`var'.gph,replace);
};

graph combine `shock'_lrgdp.gph `shock'_lxtot.gph `shock'_lrstockp.gph 
   `shock'_ltothours.gph `shock'_lrcons.gph `shock'_lrnri.gph;

capture log close;
