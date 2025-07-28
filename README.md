# Drone Gesture Recognition Control

Using a webcam feed to detect hand gestures in real time and translating them into control signals for a drone. Leverages MediaPipe to track hand landmarks and classify gestures. Uses debouncing to prevent command flickering.

## Setup
Install requirements.txt, run venv via `source venv/bin/activate`, then run `python main.py`

## Commands
- `q`: quit program
- `d`: debug info