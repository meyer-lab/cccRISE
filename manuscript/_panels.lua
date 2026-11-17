-- Figure panel references, e.g. {{< fig 2 cp-factor-matrices.2 >}} -> "Fig. 2f".
-- Args: a main figure number or supplement name, then a panel module (its i-th
-- axes with `.i`), and optionally a second module ending a range. Hyphens
-- stand in for module names' underscores. Letters come from _panels.yml
-- (written by cccrise-fbuild), supplement labels from their position in
-- `ev_figures` or `appendix_figures`, tables' from `appendix_tables`, and
-- reviewer figures' (the response only) from `reviewer_figures`.
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

-- MSB's classes of supplementary figure, and the response's reviewer
-- figures: the metadata list naming each class's figures in order, its label
-- in references and legend titles, and its figures' prefix in _panels.yml
-- (figS_<name>, figR_<name>)
local CLASSES = {
  { list = "ev_figures", ref = "Fig. EV", title = "Figure EV", prefix = "S_" },
  { list = "appendix_figures", ref = "Appendix Fig. S", title = "Appendix Figure S", prefix = "S_" },
  { list = "appendix_tables", ref = "Appendix Table S", title = "Appendix Table S", prefix = "S_" },
  { list = "reviewer_figures", ref = "Reviewer Fig. R", title = "Reviewer Figure R", prefix = "R_" },
}

-- A supplement's reference label, legend title and _panels.yml prefix, e.g.
-- "Fig. EV1", "Figure EV1", "S_"
local function supplement(name, meta)
  for _, class in ipairs(CLASSES) do
    for i, listed in ipairs(meta[class.list] or {}) do
      if str(listed) == name then return class.ref .. i, class.title .. i, class.prefix end
    end
  end
  quarto.log.warning("No supplementary figure or table class for " .. name)
  return "[" .. name .. "]", "[" .. name .. "]", "S_"
end

-- The figure's label and the letters of panels `args[2]` to `args[3]`
local function resolve(args, meta)
  local figure = str(args[1]):gsub("-", "_")
  local label = "Fig. " .. figure
  if not figure:match("^%d+$") then
    local _, prefix
    label, _, prefix = supplement(figure, meta)
    figure = prefix .. figure
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
  -- "Appendix Table S1": a table reference in the text
  tbl = function(args, _, meta)
    return pandoc.Str((supplement(str(args[1]):gsub("-", "_"), meta)))
  end,
  -- "f": panel letters alone, for captions
  panel = function(args, _, meta)
    local _, panels = resolve(args, meta)
    return pandoc.Str(panels)
  end,
  -- "Figure EV1", "Appendix Figure S1", "Reviewer Figure R1": a legend title
  label = function(args, _, meta)
    local _, title = supplement(str(args[1]):gsub("-", "_"), meta)
    return pandoc.Str(title)
  end,
}
