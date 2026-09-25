.PHONY: clean test ty profile_cpu manuscript

flist = $(wildcard cccrise/figures/figure*.py)
allOutput = $(patsubst cccrise/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

output/figure%.svg: cccrise/figures/figure%.py
	@ mkdir -p ./output
	uv run cccrise-fbuild $*

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

manuscript:
	quarto render manuscript

clean:
	rm -rf output profile profile.svg
	rm -rf factor_cache
	rm -rf manuscript/_output manuscript/.quarto
