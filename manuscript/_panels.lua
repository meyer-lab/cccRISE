-- Figure panel references, e.g. {{< fig 2 cp-factor-matrices.2 >}} -> "Fig. 2f".
-- Args: a main figure number or supplement name, then a panel module (its i-th
-- axes with `.i`), and optionally a second module ending a range. Hyphens
-- stand in for module names' underscores. Letters come from _panels.yml
-- (written by cccrise-fbuild), supplement numbers from `supplements`.
local str = pandoc.utils.stringify

local function letters(panels, spec)
  local module, index = spec:match("^([^.]+)%.?(%d*)$")
  local ls = panels[(module:gsub("-", "_"))]
  if not ls then
    quarto.log.warning("No panel letters for " .. spec)
    return { "??" }
  end
  ls = ls:map(str)
  return index == "" and ls or { ls[tonumber(index)] }
end

-- The figure's label and the letters of panels `args[2]` to `args[3]`
local function resolve(args, meta)
  local figure = str(args[1]):gsub("-", "_")
  local label = "Fig. " .. figure
  if not figure:match("^%d+$") then
    local number = "[" .. figure .. "]"
    for i, name in ipairs(meta.supplements or {}) do
      if str(name) == figure then number = tostring(i) end
    end
    label, figure = "Supplementary Fig. " .. number, "S_" .. figure
  end
  if #args < 2 then return label, "" end
  local panels = meta["fig" .. figure] or {}
  local first = letters(panels, str(args[2]))[1]
  local last = letters(panels, str(args[#args]))
  last = last[#last]
  return label, first == last and first or first .. "–" .. last
end

return {
  -- "Fig. 2f", "Supplementary Fig. 4a–b": references in the text
  fig = function(args, _, meta)
    local label, panels = resolve(args, meta)
    return pandoc.Str(label .. panels)
  end,
  -- "f": panel letters alone, for captions
  panel = function(args, _, meta)
    local _, panels = resolve(args, meta)
    return pandoc.Str(panels)
  end,
}
