@echo off
setlocal enabledelayedexpansion

rem Set input and output file names
set "base_name=Introduction"

set "input_lyx=%base_name%.lyx"
set "input=%base_name%.tex"
rem set "output=%base_name%_handout.tex"
set "output_no_draft=%base_name%.tex"
set "output_handout=%base_name%_handout.tex"
set "temp_file=%base_name%_temp.tex"


start /wait lyx --export latex "%input_lyx%"


rem Second Pass: Add 'handout' option to \documentclass line
if exist "%output_handout%" del "%output_handout%"

for /f "tokens=*" %%A in ('type "%input%"') do (
    set "line=%%A"
    
    rem Enable delayed expansion inside the loop to update the line variable
    setlocal enabledelayedexpansion
    
    rem Check if the line contains \documentclass and replace only if it does
    if "!line!"=="\documentclass[10pt,english,t,10pt]{beamer}" (
        set "line=\documentclass[10pt,english,t,10pt,handout]{beamer}"
    )
    
    rem Append the modified line to the final output file
    rem Avoid echoing empty lines
    if not "!line!"=="" (
        echo !line!>> "%output_handout%"
    )
    
    endlocal
)


rem rem Loop through each line in the input file
rem for /f "tokens=*" %%A in ('type "%input%"') do (
rem     set "line=%%A"
    
rem     rem Enable delayed expansion inside the loop to update the line variable
rem     setlocal enabledelayedexpansion
    
rem     rem Replace \documentclass{beamer} with \documentclass[handout]{beamer}
rem     set "line=!line:\documentclass[10pt,english,t,10pt]{beamer}=\documentclass[10pt,english,t,10pt,handout]{beamer}!"
    
rem     rem remove draft option 
    
rem     set "line=!line:draft,=!"

rem     rem Append the modified line to the output file
rem     echo !line!>> "%output%"
    
rem     endlocal
rem )

rem pdflatex "%input%"
pdflatex "%input%"
pdflatex "%output_handout%"


rem copy 
set "parent_dir=.."

set "output_pdf_parent=%parent_dir%\%base_name%.pdf"
copy /y "%base_name%.pdf" "%output_pdf_parent%"

set "output_pdf_parent=%parent_dir%\%base_name%_handout.pdf"
copy /y "%base_name%_handout.pdf" "%output_pdf_parent%"


rem clean up 
del "*.aux" "*.log" "*.nav" "*.out" "*.snm" "*.toc"

del "%base_name%.pdf"
del "%base_name%_handout.pdf"