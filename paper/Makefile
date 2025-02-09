# Variables
INPUT_MD = culturally_sensitive_agents.md
OUTPUT_TEX = culturally_sensitive_agents.tex
OUTPUT_PDF = culturally_sensitive_agents.pdf

# Default target
all: pdf

# Convert markdown to tex
tex: $(INPUT_MD)
	pandoc -F mermaid-filter \
		--from markdown+raw_html \
		--wrap=auto \
		--columns=72 \
		--standalone \
		--preserve-tabs \
		-o $(OUTPUT_TEX) $(INPUT_MD)

# Convert markdown to pdf
pdf: $(INPUT_MD)
	pandoc -F mermaid-filter \
		--from markdown+raw_html \
		--wrap=auto \
		--columns=72 \
		--standalone \
		--preserve-tabs \
		--pdf-engine=xelatex \
		--variable=geometry:margin=1in \
		--variable=linestretch=1.15 \
		--variable=fontsize=11pt \
		-o $(OUTPUT_PDF) $(INPUT_MD)

# Convert latex to pdf
latexpdf: $(OUTPUT_TEX)
	xelatex $(OUTPUT_TEX)
	xelatex $(OUTPUT_TEX)  # Second run for references/TOC

# Clean generated files
clean:
	rm -f $(OUTPUT_TEX) $(OUTPUT_PDF) *.aux *.log *.out *.toc

.PHONY: all tex pdf latexpdf clean