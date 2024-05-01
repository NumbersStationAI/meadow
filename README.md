# Meadow
Framework for building data agent workflows

> I like to think (and the sooner the better!) of a cybernetic meadow where mammals and computers live together in mutually programming harmony like pure water touching clear sky.
>
> --Richard Brautigan

<p align="center">
<img src="assets/meadow_image.png" alt="Cybernetic Meadow"/>
</p>

# Setup
## Install
### Poetry
If you don't have `poetry` installed, follow instructions from [here](https://python-poetry.org/docs/#installing-with-the-official-installer). We recommend

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Meadow
Then install Meadow with
```
cd meadow
make install
```

### DuckDB (optional)
If you want to load some sample data in `examples/data`, you will need `duckdb`. To get the executable, go to [here](https://duckdb.org/docs/installation). Then run the following. You can replace the filename with anything you want.

```bash
cd meadow/examples/data
duckdb sales_example.duckdb
.read sales.sql
.exit
```

# Demo Run
To get started with a simple usecase using text2sql on a duckdb database, we have `examples/demo.py` to run. You will need an Anthropic API key.
```bash
poetry run python examples/demo.py \
  --api-key API_KEY \
  --duckdb-file examples/data/sales_example.duckdb \
  --instruction Find\ the\ top\ 5\ most\ popular\ products\ fro
m\ the\ customers\ who\ order\ the\ most\ pantry\ items.
```
If you don't give an instruction, you will be prompted for one at the start.

# Meadow Framework
TBD