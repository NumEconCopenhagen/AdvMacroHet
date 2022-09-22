********************************************************************************
* Generate graph
********************************************************************************
* Note - the #N/A's that appear in the graph in the paper need to be added in
* after running this code using the Stata graph editor
********************************************************************************
clear
#delimit;
* import;
import excel "ParkerExperiment.xlsx", sheet("Sheet1") cellrange(A3:F9) firstrow clear;
*replace DataParker=0 if DataParker==.;
gen empty =.;

graph bar LiquidAssetsFrictionless LiquidAssetsStickyExpectat empty DataParker, over(Quarter) legend(pos(1) ring(0) col(1) order(1 2 4)) yvaroptions( relabel(1 "Frictionless Expectations" 2 "Sticky Expectations" 4 "Data")) 
ytitle("Percent of Payment Spent in Quarter", height(10)) b1title("Quarter Relative to Receipt of Payment") blabel(total)
bar(1, color(blue*0.4)) bar(2, color(blue*1.2))  bar(4, color(blue*5)) graphregion(color(white)); 

graph export "fParker.pdf", replace;
graph export "fParker.wmf", replace;
graph export "fParker.png", replace;
graph export "fParker.svg", replace;
