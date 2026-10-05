-- Turn ::: callout divs into tcolorbox boxes in LaTeX output.
function Div(el)
  if el.classes:includes('callout') and FORMAT:match('latex') then
    local title = el.attributes.title or 'What you need to know'
    table.insert(el.content, 1, pandoc.RawBlock('latex', '\\begin{tcolorbox}[colback=DarkRed!4,colframe=DarkRed,fonttitle=\\bfseries,title={' .. title .. '}]'))
    table.insert(el.content, pandoc.RawBlock('latex', '\\end{tcolorbox}'))
    return el.content
  end
end
