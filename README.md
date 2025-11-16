# Urban Noise Cancellation System

### Active Noise Reduction in a 2D Acoustic Field Using Multi-Mic Recording and Dual-Speaker Output

This project focuses on building an urban noise cancellation system that captures surrounding environmental noise using microphones placed in a square arrangement and reduces unwanted external sound using two speakers. The system aims to model, analyze, and cancel noise within a defined 2D area using DSP, filtering, and adaptive learning techniques.

## Project Overview

Modern urban environments are filled with unpredictable noise—traffic, construction, crowds, and wind. This project attempts to create a controlled quiet zone by:

- Recording audio using multiple microphones positioned in a square.
- Analyzing noise patterns (frequency, amplitude, direction).
- Generating inverse sound waves using two speakers to cancel out external noise.
- Optimizing cancellation using DSP techniques and optionally ML-driven models.

This system is inspired by active noise cancellation (ANC) used in headphones, but extended to a spatial environment.

```
 ┌───────────────────────────┐
 │   Microphone Array (xn)   │
 │  [Top-Left][Top-Right]    │
 │  [Bot-Left][Bot-Right]    │
 └───────────┬────────────-──┘
             │ Captured Noise
             ▼
 ┌──────────────────────────----─┐
 │   Signal Processing Core      │
 │ - Preprocessing (filtering)   |
 │ - Cross-correlation           |
 │ - Direction of Arrival (DoA)  |
 │ - Adaptive filtering (LMS)    |
 │ - Anti-noise generation       |
 └───────────┬──────────────-----┘
             │ Anti-Noise Waveform
             ▼
 ┌───────────────────────────┐
 │     Speakers (x2)         │
 │  Emit cancellation signal │
 └───────────────────────────┘

```


## Note
This Repo contains the supporting code files for the Urban Noise Cancellation System. It DOES NOT CONTAIN the whole code.
