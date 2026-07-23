Flood Data
├── events_master.csv        every event: event_id, batch, n_chips, productive,
│                              + all Stage-A provenance (bbox, dates, continent,
│                              country, tier, corroboration, gfm_flood_km2 …)
├── chips/<event_id>/          one folder per event, named by its assigned ID
│      ├── <chip_id>_label.tif
│      ├── <chip_id>_s2_pre/_s2_post/_s1_pre/_s1_post.tif
│      └── <chip_id>.json    (dates, clear_frac, provenance, place)
├── checkpoints/              spot-check figures every ~50 events (~36 figures)
└── report/                   REPORT.md + tables/ + global event map
