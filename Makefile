.PHONY: install demo

install:
	poetry install

demo:
	poetry run streamlit run demo_app.py

