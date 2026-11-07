.PHONY: clean test ty profile_cpu manuscript source_data source_data_check tables

# One package per figure (cccrise/figures/figure<N>/; a supplement is nested as
# figure<N>/figureS_<name>/, or top-level if it supports no one figure), built
# from its master figure<N>.py into output/figure<N>/figure<N>.svg. Build one
# panel with `uv run cccrise-fbuild <N> <panel>`.
fpkgs = $(wildcard cccrise/figures/figure*/) $(wildcard cccrise/figures/figure*/figureS_*/)
fdirs = $(foreach p,$(fpkgs),$(notdir $(p:/=)))
allOutput = $(foreach d,$(fdirs),output/$(d)/$(d).svg)
figureDeps = cccrise/figures/common.py cccrise/figures/style.py
# A figure's package, and the parent package whose _common.py it shares
fpkg = $(filter %/$(1)/,$(fpkgs))
fparent = $(dir $(patsubst %/,%,$(call fpkg,$(1))))

all: $(allOutput)

.SECONDEXPANSION:
output/%.svg: $$(wildcard $$(call fpkg,$$(notdir $$*))*.py $$(call fparent,$$(notdir $$*))_common.py) $(figureDeps)
	uv run cccrise-fbuild $(patsubst figure%,%,$(notdir $*))

# EMBO source data: one zip per figure in output/source_data/, from the panel
# data each figure's build saves, plus docs/source_data_manifest.md
source_data: all
	uv run cccrise-sourcedata

# Rebuild every figure from scratch and check its source data is unchanged
source_data_check: source_data
	rm -rf output/source_data_previous
	cp -r output/source_data output/source_data_previous
	$(MAKE) -B source_data
	diff -r -x '*.zip' output/source_data_previous output/source_data
	@echo "Source data reproduced exactly"

profile-cpu: .venv
	@ echo "Profiling figure $(F)..."
	@ # Step 1: Run the profiler and save stats to a file. This will still show errors if the script crashes.
	uv run python -m cProfile -o ./output/profile_cpu_$(F).prof .venv/bin/cccrise-fbuild $(F)
	@ echo "\n--- Profiler Results ---"
	@ # Step 2: Print the saved stats, sorted by total time.
	uv run python -c "import pstats; p = pstats.Stats('./output/profile_cpu_$(F).prof'); p.sort_stats('tottime').print_stats(100)"

test: .venv
	uv run pytest -s -v -x

.venv:
	uv sync

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=cccrise --cov-report xml:coverage.xml

ty: .venv
	uv run ty check cccrise

# Appendix tables, as Markdown in output/tables/, included by the Appendix
tables:
	uv run python -m cccrise.figures.tables

manuscript: tables
	quarto render manuscript

clean:
	rm -rf output profile profile.svg
	rm -rf factor_cache
	rm -rf manuscript/_output manuscript/.quarto
