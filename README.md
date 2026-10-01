# ArcSentinel

**FPGA-accelerated detection of downed and arcing power lines (high-impedance faults)**
Team ArcSentinel: Hemanth Chilakala (team lead), Tanmaya Kumar
PolarFire® FPGA Design Contest 2026 · **Track 2: PolarFire SoC Icicle Kit (MPFS250T)**

When a live distribution line falls onto dry grass, sand or asphalt, it often draws too little current
for a conventional relay to notice. The line stays energised and arcs, which can start wildfires and
electrocute people. ArcSentinel runs a deterministic DSP + int8-AI pipeline in PolarFire SoC fabric on
every power cycle. A bare-metal RISC-V hart makes the trip decision, and Linux reports it over redundant
IEC 61850 networks (PRP on both Gigabit Ethernet ports).

| Pre-proposal model (synthetic data, held-out set) | ArcSentinel (int8) | 51N + inrush restraint |
|---|---|---|
| High-impedance faults detected (120 events, 0.4–40 A rms arc current) | **98.3%** | 2.5% |
| … of which sub-1 A arcs (24 events) | **91.7%** | 0% |
| Look-alike events with no false operation (720 events) | **100%** | 98.6% |
| Median detection time after onset | **0.28 s** | n/a |

## Proposal documents

| File | Content |
|---|---|
| [`docs/ArcSentinel_System_Block_Diagram.pdf`](docs/ArcSentinel_System_Block_Diagram.pdf) | Required block diagram: system architecture, data path, mixed-criticality partition |
| [`docs/ArcSentinel_Technical_Brief.pdf`](docs/ArcSentinel_Technical_Brief.pdf) | Feasibility evidence, IP cores, resource estimate, test plan, schedule, risks |
| [`portal_text/`](portal_text) | Project Overview (≤300 words) and Innovation & Impact (≤250 words), as entered on the portal |
| [`feasibility/`](feasibility) | Python golden model and results used in the proposal |

## Reproduce the feasibility study

```bash
pip install numpy matplotlib
python feasibility/arcsentinel_model.py
```

Runtime is about 1 minute on a laptop. It writes `feasibility/results/results.json` and three figures.
The results are deterministic (fixed seeds). The training and test sets come from independent seeds.

What the model contains, and how each part maps to hardware:

| Stage | Golden model | Planned hardware |
|---|---|---|
| Waveforms | Emanuel arc HIF model + 6 look-alike classes, 25.6 kS/s | HIL replay (LPDDR4 injector / DAC) |
| Per-cycle features | FFT harmonics, Haar DWT, asymmetry, cycle-difference, h3 angle | Fabric: CoreFFT + SmartHLS |
| Window statistics | mean / σ over 10 cycles, hop 2 | Fabric: SmartHLS |
| Classifier | MLP 28-32-16-7, int8 weights and activations, int32 accumulate | Fabric: SmartHLS on Math blocks |
| Security logic | 5 consecutive HIF windows | Bare-metal U54 hart |
| Baseline | 51N definite-time, 30 A pickup, 2nd-harmonic inrush restraint | n/a |

> All data are synthetic, so these figures are an upper bound: HIF detectors typically score lower on
> field data. The next step is validation on the public Victorian *Vegetation Conduction Ignition Test*
> fault database (CC BY 4.0).

**Prior art.** HIF detection itself exists (SEL Arc Sense, GE Hi-Z, and UNSW's patented FPGA + CNN
edge device, US 12,032,037 B2). ArcSentinel's contribution is the three-layer mixed-criticality
architecture on one chip, all-fabric detection at fixed latency, PRP redundancy and replay self-test.
See Technical Brief §2.

## Repository contents (proposal stage)

```
docs/
  ArcSentinel_System_Block_Diagram.pdf   required block diagram (2 pages)
  ArcSentinel_Technical_Brief.pdf        feasibility, prior art, plan, risks (7 pages)
  block_diagram.html, technical_brief.html   sources for the two PDFs
feasibility/
  arcsentinel_model.py                   Python golden model (synthetic data)
  requirements.txt                       numpy, matplotlib
  results/                               results.json, run log, figures
portal_text/                             text entered in the proposal portal
```

## Planned repository layout (development phase)

```
fabric/      Libero SoC project, RTL (adc_seq, replay_inj, ocp_50_51, trip_logic), SmartHLS C++ cores
firmware/    bare-metal protection hart (context B), RPMsg
linux/       Yocto/Buildroot AMP layer, IEC 61850 GOOSE/MMS app, PRP/PTP config, web HMI
hil/         waveform library generator, COMTRADE tools, test harness
feasibility/ golden model (this stage)
docs/        proposal, final report, diagrams
```

## Disclosure

Third-party IP and open-source components are listed in the Technical Brief, section 11.
AI assistance: Claude (Anthropic) was used to help draft proposal text, lay out the diagrams and
scaffold the Python feasibility model. The team reviewed and ran all code, checked all results, and
owns the design.

## License

Code and documents in this repository are released under the MIT License (see `LICENSE`).
Third-party components keep their own licences.
