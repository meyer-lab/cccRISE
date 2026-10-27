-- Bold the "Figure N." caption prefix together with the bold caption title.
-- Runs post-render, once Quarto has inserted the prefix into the caption.
local function bold_prefix(block)
  local inlines = block.content
  local start
  for i, el in ipairs(inlines) do
    if el.t == "Str" and el.text == "Figure" then start = i break end
  end
  if not start then return end
  local stop = start
  while inlines[stop + 1] and inlines[stop + 1].t ~= "Space" do stop = stop + 1 end
  local prefix = pandoc.List({ table.unpack(inlines, start, stop) })
  local title = inlines[stop + 2]
  if title and title.t == "Strong" then
    title.content = prefix .. { pandoc.Space() } .. title.content
    for _ = start, stop + 1 do inlines:remove(start) end
  else
    for _ = start, stop do inlines:remove(start) end
    inlines:insert(start, pandoc.Strong(prefix))
  end
  return block
end

function Div(el)
  if el.identifier:match("^fig%-") then
    return el:walk({ Plain = bold_prefix, Para = bold_prefix })
  end
end
