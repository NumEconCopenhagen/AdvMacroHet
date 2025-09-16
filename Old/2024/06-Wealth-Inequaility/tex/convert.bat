@echo off
setlocal enabledelayedexpansion

rem Set input and output file names
set "base_name=Wealth-Inequality"

set "input_lyx=%base_name%.lyx"
set "input=%base_name%.tex"
rem set "output=%base_name%_handout.tex"
set "output_no_draft=%base_name%.tex"
set "output_handout=%base_name%_handout.tex"
set "temp_file=%base_name%_temp.tex"


start /wait lyx --export latex "%input_lyx%"


rem Clear the temporary output file if it exists
if exist "%temp_output%" del "%temp_output%"

rem Use the findstr command to replace text in a single command
(
    for /f "usebackq delims=" %%A in ("%input%") do (
        set "line=%%A"
        
        rem Check if the line contains the specific string and replace if it does
        set "line=!line:\documentclass[10pt,english,t,10pt]{beamer}=\documentclass[10pt,english,t,10pt,handout]{beamer}!"
        
        rem Append the modified line to the temporary output file
        echo !line! >> "%temp_output%"
    )
)

rem Replace the original file with the modified content
move /y "%temp_output%" "%input%"
endlocal

rem rem Second Pass: Add 'handout' option to \documentclass line
rem if exist "%output_handout%" del "%output_handout%"

rem for /f "tokens=*" %%A in ('type "%input%"') do (
rem     set "line=%%A"
    
rem     rem Enable delayed expansion inside the loop to update the line variable
rem     setlocal enabledelayedexpansion
    
rem     rem Check if the line contains \documentclass and replace only if it does
rem     if "!line!"=="\documentclass[10pt,english,t,10pt]{beamer}" (
rem         set "line=\documentclass[10pt,english,t,10pt,handout]{beamer}"
rem     )
    
rem     rem Append the modified line to the final output file
rem     rem Avoid echoing empty lines
rem     if not "!line!"=="" (
rem         echo !line!>> "%output_handout%"
rem     )
    
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