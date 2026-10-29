-- Figure panel references, e.g. {{< fig 2 cp-factor-matrices.2 >}} -> "Fig. 2f".
-- Args: a main figure number or supplement name, then a panel module (its i-th
-- axes with `.i`), and optionally a second module ending a range. Hyphens
-- stand in for module names' underscores. Letters come from _panels.yml
-- (written by cccrise-fbuild), supplement labels from their position in
-- `ev_figures` or `appendix_figures`.
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

-- MSB's classes of supplementary figure: the metadata list naming each
-- class's figures in order, and its label in references and legend titles
local CLASSES = {
  { list = "ev_figures", ref = "Fig. EV", title = "Figure EV" },
  { list = "appendix_figures", ref = "Appendix Fig. S", title = "Appendix Figure S" },
}

-- A supplement's reference label and legend title, e.g. "Fig. EV1", "Figure EV1"
local function supplement(name, meta)
  for _, class in ipairs(CLASSES) do
    for i, listed in ipairs(meta[class.list] or {}) do
      if str(listed) == name then return class.ref .. i, class.title .. i end
    end
  end
  quarto.log.warning("No supplementary figure class for " .. name)
  return "[" .. name .. "]", "[" .. name .. "]"
end

-- The figure's label and the letters of panels `args[2]` to `args[3]`
local function resolve(args, meta)
  local figure = str(args[1]):gsub("-", "_")
  local label = "Fig. " .. figure
  if not figure:match("^%d+$") then
    label, figure = supplement(figure, meta), "S_" .. figure
  end
  if #args < 2 then return label, "" end
  local panels = meta["fig" .. figure] or {}
  local first = letters(panels, str(args[2]))[1]
  local last = letters(panels, str(args[#args]))
  last = last[#last]
  return label, first == last and first or first .. "–" .. last
end

return {
  -- "Fig. 2f", "Fig. EV2a–b", "Appendix Fig. S4a": references in the text
  fig = function(args, _, meta)
    local label, panels = resolve(args, meta)
    return pandoc.Str(label .. panels)
  end,
  -- "f": panel letters alone, for captions
  panel = function(args, _, meta)
    local _, panels = resolve(args, meta)
    return pandoc.Str(panels)
  end,
  -- "Figure EV1", "Appendix Figure S1": a supplement's legend title
  label = function(args, _, meta)
    local _, title = supplement(str(args[1]):gsub("-", "_"), meta)
    return pandoc.Str(title)
  end,
}
