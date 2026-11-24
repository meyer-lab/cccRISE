-- A spreadsheet's table, e.g. {{< xlsx figures/tables/alad_outcome_groups.xlsx >}}:
-- the first sheet's run of rows from the first full-width row (the header)
-- to the next blank row, leaving the title and notes around it to the
-- legend. Merged cells span their rows and columns, cell indents become em
-- spaces, and all-number columns are right aligned. The PDF also carries
-- the workbook as an attachment.
local EM_SPACE = "\u{2003}"

local function unescape(s)
  return (s:gsub("&lt;", "<"):gsub("&gt;", ">"):gsub("&quot;", '"'):gsub("&apos;", "'"):gsub("&amp;", "&"))
end

-- A string item's text: its <t> runs joined (Excel splits rich text into runs)
local function text(xml)
  local parts = {}
  for t in xml:gmatch("<t[^>]*>(.-)</t>") do parts[#parts + 1] = unescape(t) end
  return table.concat(parts)
end

-- Rewrite self-closing <tag .../> as <tag ...></tag> so one pattern reads both
local function open_close(xml, tag)
  return (xml:gsub("<" .. tag .. "([%s][^>]-)/>", "<" .. tag .. "%1></" .. tag .. ">"))
end

-- Row and column numbers of a cell reference such as "B4"
local function position(ref)
  local letters, r = ref:match("(%u+)(%d+)")
  local col = 0
  for letter in letters:gmatch(".") do col = col * 26 + letter:byte() - 64 end
  return tonumber(r), col
end

local function unzip(path)
  local f = assert(io.open(path, "rb"))
  local files = {}
  for _, entry in ipairs(pandoc.zip.Archive(f:read("a")).entries) do
    files[entry.path] = entry:contents()
  end
  f:close()
  return files
end

-- The first sheet's {row: {column: text}} and {row: {column: {rows, columns}}}
-- spans of merged cells, with the cells they cover marked false
local function read_sheet(files)
  local shared = {}
  for si in (files["xl/sharedStrings.xml"] or ""):gmatch("<si>(.-)</si>") do shared[#shared + 1] = text(si) end
  local indents = {}
  local xfs = open_close(files["xl/styles.xml"]:match("<cellXfs.->(.-)</cellXfs>"), "xf")
  for xf in xfs:gmatch("<xf.->(.-)</xf>") do indents[#indents + 1] = tonumber(xf:match('indent="(%d+)"')) or 0 end
  local sheet = open_close(files["xl/worksheets/sheet1.xml"], "c")
  local rows = {}
  for body in sheet:gmatch("<row[%s][^>]->(.-)</row>") do
    for attrs, inner in body:gmatch("<c[%s](.-)>(.-)</c>") do
      local r, col = position(attrs:match('r="(%u+%d+)"'))
      local value = attrs:match('t="s"') and shared[tonumber(inner:match("<v>(.-)</v>")) + 1]
        or attrs:match('t="inlineStr"') and text(inner)
        or unescape(inner:match("<v>(.-)</v>") or "")
      rows[r] = rows[r] or {}
      if value ~= "" then
        rows[r][col] = EM_SPACE:rep(2 * indents[(tonumber(attrs:match('s="(%d+)"')) or 0) + 1]) .. value
      end
    end
  end
  local spans = {}
  for first, last in sheet:gmatch('<mergeCell ref="(%u+%d+):(%u+%d+)"') do
    local r1, c1 = position(first)
    local r2, c2 = position(last)
    for r = r1, r2 do
      rows[r] = rows[r] or {}
      for c = c1, c2 do rows[r][c] = (r == r1 and c == c1) and (rows[r][c] or "") or false end
    end
    spans[r1] = spans[r1] or {}
    spans[r1][c1] = { r2 - r1 + 1, c2 - c1 + 1 }
  end
  return rows, spans
end

-- Row numbers and column range of the sheet's table: from the first row
-- filling every used column (merge-covered cells count as filled) to the
-- row before the next blank one
local function table_extent(rows)
  local c1, c2, last = math.huge, 0, 0
  for r, row in pairs(rows) do
    last = math.max(last, r)
    for c, value in pairs(row) do
      if value ~= "" then c1, c2 = math.min(c1, c), math.max(c2, c) end
    end
  end
  local function full(r)
    for c = c1, c2 do if (rows[r] or {})[c] == nil then return false end end
    return true
  end
  local function blank(r)
    for c = c1, c2 do if (rows[r] or {})[c] ~= nil then return false end end
    return true
  end
  local first = 1
  while first <= last and not full(first) do first = first + 1 end
  local numbers = {}
  for r = first, last do
    if blank(r) then break end
    numbers[#numbers + 1] = r
  end
  return numbers, c1, c2
end

local function to_table(rows, spans)
  local numbers, c1, c2 = table_extent(rows)
  local colspecs = {}
  for c = c1, c2 do
    local numeric = true
    for i = 2, #numbers do
      local value = rows[numbers[i]][c]
      numeric = numeric and (not value or value == "" or tonumber(value) ~= nil)
    end
    colspecs[#colspecs + 1] = { numeric and "AlignRight" or "AlignLeft" }
  end
  local function row(r)
    local cells = {}
    for c = c1, c2 do
      local value = rows[r][c]
      if value ~= false then
        local span = (spans[r] or {})[c] or { 1, 1 }
        cells[#cells + 1] = pandoc.Cell({ pandoc.Plain({ pandoc.Str(value or "") }) }, "AlignDefault", span[1], span[2])
      end
    end
    return pandoc.Row(cells)
  end
  local body = {}
  for i = 2, #numbers do body[#body + 1] = row(numbers[i]) end
  return pandoc.Table(
    { long = {}, short = {} },
    colspecs,
    pandoc.TableHead({ row(numbers[1]) }),
    { { attr = pandoc.Attr(), body = body, head = {}, row_head_columns = 0 } },
    pandoc.TableFoot()
  )
end

return {
  xlsx = function(args)
    local path = pandoc.utils.stringify(args[1])
    local blocks = { to_table(read_sheet(unzip(path))) }
    if quarto.doc.is_format("typst") then
      blocks[#blocks + 1] = pandoc.RawBlock("typst", '#pdf.attach("' .. path .. '")')
    end
    return blocks
  end,
}
