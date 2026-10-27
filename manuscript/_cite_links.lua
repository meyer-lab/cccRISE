-- Run citeproc here (instead of at the end of the pandoc pipeline) so citation
-- links can be pointed at the cited work instead of the in-page reference list.
local PREFIX_URLS = {
  doi = "https://doi.org/",
  pmid = "https://pubmed.ncbi.nlm.nih.gov/",
  arxiv = "https://arxiv.org/abs/",
}

local function external_link(el)
  local prefix, id = el.target:match("^#ref%-(%a+):(.+)$")
  if prefix and PREFIX_URLS[prefix] then
    el.target = PREFIX_URLS[prefix] .. id
    return el
  end
end

function Pandoc(doc)
  return pandoc.utils.citeproc(doc):walk({ Link = external_link })
end
