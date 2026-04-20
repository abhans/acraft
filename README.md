# The Environment

To run the application, use the command below:

```bash
python -W ignore -m src.main
```

## Collecting Expert Data

For GAIL, expert data can be collected with running the `collector.py` as a standalone script:

```bash
python -W ignore -m core.collector
```

## Animating the Expert Data

To validate the expert data, it can be animated using `animate.py` script:

```bash
python -W ignore -m data.animate
```
