SOURCES = $(wildcard plots/*.png)
TARGETS = $(patsubst plots/%.png, plots/cropped/%.png, $(SOURCES))


.PHONY: all clean
all: $(TARGETS)


plots/cropped/%.png: plots/%.png
	mkdir -p plots/cropped
	magick $< -trim +repage -background white $@


clean:
	rm -rf plots/cropped
	