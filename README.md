# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bigD90girth/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bigD90girth/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players will hold each PlayStation 5 trophy a week after launch, from Exophase's earn rates. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one".

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case in the input directory and writes `<id>.out` for each: five quantiles (5%, 25%, 50%, 75%, 95%) of each trophy's earn rate at the case's target, in percent.

## Model

Per trophy, in logit space: a least-squares fit of logit(earn rate) against ln(1 + hours since release), on the last 36 hours of reads, weighted toward recent reads and shrunk toward the trophy tier's median slope. The median is the last read moved by that slope to the target; the spread combines a floor for Exophase's refresh jumps, a random-walk term growing with the square root of the horizon, the fit's residuals, and the slope's uncertainty. Standard library only, deterministic, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=kk8gvjx9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
