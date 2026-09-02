# NBA Live Win Probability

A machine learning project to estimate NBA home-team win probability from play-by-play game state data.

## Planned Stack

- Python
- nba_api
- pandas
- PyTorch
- Flask
- Flask-SocketIO

## Project Goals

- Collect historical NBA play-by-play data
- Build game-state training examples
- Train a win probability model
- Serve predictions through an API
- Display live or replayed probabilities on a dashboard

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt