.PHONY: install crawl index run docs clean help

help:
	@echo "Cloud.ru AI-Репетитор"
	@echo ""
	@echo "Основные команды:"
	@echo "  make install  - установка зависимостей"
	@echo "  make crawl    - парсинг документации"
	@echo "  make index    - создание эмбеддингов"
	@echo "  make run      - запуск интерфейса"
	@echo ""
	@echo "Документация:"
	@echo "  make docs     - сборка презентации (reveal.js + PDF)"
	@echo "  make docs-html - только HTML (reveal.js)"
	@echo "  make docs-pdf  - только PDF (beamer)"
	@echo "  make clean    - очистка временных файлов"

install:
	uv sync

crawl:
	uv run python crawl.py

index:
	uv run python embeddings.py

run:
	uv run streamlit run app.py

docs: docs-diagrams docs-html docs-pdf

docs-diagrams:
	@echo "Генерация PlantUML диаграмм..."
	@for f in assets/*.puml; do \
		echo "  $$f"; \
		plantuml -tpng "$$f"; \
	done

docs-html:
	@echo "Сборка презентации (reveal.js)..."
	pandoc PRESENTATION.md -t revealjs -s -o presentation.html \
		--css=assets/presentation.css \
		-V theme=white \
		-V transition=slide \
		-V slideNumber=true \
		-V hash=true

docs-pdf:
	@echo "Сборка презентации (PDF)..."
	pandoc PRESENTATION.md -d beamer -o presentation.pdf

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .ruff_cache .pytest_cache
	rm -f *.log
