/* Paths are relative to docs/index.html. Empty src retains a placeholder. */
window.ALTER_MEDIA = {
  "presentation": {
    "type": "video",
    "src": "videos/presentation.mp4",
    "alt": "ALTER project presentation"
  },
  "overview": {
    "type": "image",
    "src": "pictures/overview.png",
    "alt": "ALTER adapts a frozen single-arm diffusion policy using multi-agent demonstrations and policy-distilled replay, enabling decentralized coordination while retaining source-task behavior"
  },
  "architecture": {
    "type": "image",
    "src": "pictures/architecture.png",
    "alt": "Frozen base denoiser and residual coordination adapter conditioned on local visual observations"
  },
  "simulationSetup": {
    "type": "image",
    "src": "pictures/simulation-setup.png",
    "alt": "TwoArmPlaceWipe setup: tray, sponge, dirt, waiting area, and local views from both robots"
  },
  "hardwareSetup": {
    "type": "image",
    "src": "pictures/hardware-setup.png",
    "alt": "Hardware setup: two xArm7 robots, a bird, lid, box, and local camera views"
  },
  "results": {
    "type": "image",
    "src": "pictures/results.svg",
    "alt": "Coordination and combined source success from Tables I\u2013II. Budgets for ALTER, FS, and FT-mixed are 20/20, 40/40, and 60/60 multi-agent/distilled single-arm demonstrations; FT-multi uses 20/0, 40/0, and 60/0. ALTER coordination success is 35, 71.5, and 87 percent; its combined source success is 97, 97.5, and 97 percent.",
    "mobileSrc": "pictures/results-mobile.svg"
  },
  "simulation": {
    "type": "video",
    "src": "videos/web/simulation-coordination-01.mp4",
    "alt": "ALTER simulated coordination example 1",
    "width": 510,
    "height": 512
  },
  "source": {
    "type": "video",
    "src": "videos/web/simulation-source-01.mp4",
    "alt": "Adapted ALTER: place-and-return, example 1",
    "width": 512,
    "height": 512
  },
  "source2": {
    "type": "video",
    "src": "videos/web/simulation-source-02.mp4",
    "alt": "Adapted ALTER: place-and-return, example 2",
    "width": 512,
    "height": 512
  },
  "source3": {
    "type": "video",
    "src": "videos/web/simulation-source-03.mp4",
    "alt": "Adapted ALTER: wipe-and-return",
    "width": 512,
    "height": 512
  },
  "hardware": {
    "type": "video",
    "src": "videos/web/hardware-coordination.mp4",
    "alt": "ALTER hardware coordination",
    "width": 948,
    "height": 620
  },
  "hardwareSource": {
    "type": "video",
    "src": "videos/web/hardware-source-01.mp4",
    "alt": "Adapted ALTER: bird pick-and-place",
    "width": 1280,
    "height": 762
  },
  "hardwareSource2": {
    "type": "video",
    "src": "videos/web/hardware-source-02.mp4",
    "alt": "Adapted ALTER: lid removal",
    "width": 1280,
    "height": 810
  },
  "hardwareSource3": {
    "type": "video",
    "src": "videos/web/hardware-source-03.mp4",
    "alt": "Adapted ALTER: lid replacement",
    "width": 1280,
    "height": 802
  },
  "ftFailure": {
    "type": "video",
    "src": "videos/web/baseline-failure-01.mp4",
    "alt": "FT-mixed failure",
    "width": 1280,
    "height": 756
  },
  "fsFailure": {
    "type": "video",
    "src": "videos/web/baseline-failure-02.mp4",
    "alt": "FS failure",
    "width": 1280,
    "height": 756
  },
  "fsNearMiss": {
    "type": "video",
    "src": "videos/web/baseline-near-miss-grab.mp4",
    "alt": "FS success with nearly missed grab",
    "width": 1280,
    "height": 754
  },
  "ftNearMiss": {
    "type": "video",
    "src": "videos/web/baseline-near-miss-place.mp4",
    "alt": "FT-mixed success with nearly missed placement",
    "width": 1280,
    "height": 754
  }
};
