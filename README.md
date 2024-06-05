# Predict Stock Prices with Python
Simple application for predicting stock prices using python.  
Data collected from Yahoo finance, visualized in Streamlit.   
Predictions made with Facebook Prophet

## Setup
Clone repo
```shell
git clone git@github.com:MarkusSagen/stockpredictor-streamlit.git
cd stockpredictor-streamlit 
```

Install dependencies
```shell
poetry install 
```

Run 
```shell
poetry shell 
poetry run streamlit run main.py
```

## TODO
- [ ] Query any stock price (add more stocks) from Yahoo Finance
- [ ] Integrate or expand with `freqtrade` python API
- [ ] Add permanent storage with Postgres
- [ ] Add cashe storage with Redis
- [ ] Add users
- [ ] Extend UI to Svelte + TailwindCSS
