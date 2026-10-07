# Methods — report-ready text

Draft paragraphs for the data availability report, the updated research proposal and Paper 1.
Numbers in square brackets are placeholders to be filled from the real run. Everything else can be used as is.

## 2. Study area

The study area spans the optical gradient from inner Tokyo Bay to open Sagami Bay (139.05–140.15 °E,
34.95–35.72 °N) and is divided into five zones: inner Tokyo Bay north of the Futtsu–Kannonzaki line,
the Uraga Channel, the transition zone around the tip of the Miura Peninsula, the Shōnan coastal strip
of Sagami Bay with the Sagami and Sakawa river mouths, and open Sagami Bay. Tokyo Bay is a eutrophic,
semi-enclosed embayment with seasonal bottom hypoxia and blue-tide events; Sagami Bay is deep, open to
the Pacific and influenced by the Kuroshio. Sagami Bay is treated as a data-poor target: the question
asked of it is how few local observations are needed to adapt a model developed in Tokyo Bay.

## 3. Data

### 3.1 Satellite
PACE OCI Level-2 apparent optical properties (remote-sensing reflectance, Rrs(λ), ~184 bands,
340–890 nm, ~1.2 km) and biogeochemical products (chlorophyll-a) were obtained from the NASA Ocean
Biology DAAC for [April 2024 – September 2026]. Only the subset of each granule covering the study
area was read. GCOM-C/SGLI (250 m) and Sentinel-3 OLCI (300 m) are used for comparison and for
longer time series. Pixels were screened with the NASA standard Level-2 mask (ATMFAIL, LAND, HIGLINT,
HILT, HISATZEN, STRAYLIGHT, CLDICE, COCCOLITH, HISOLZEN, LOWLW, CHLFAIL, NAVWARN, ABSAER, MAXAERITER,
ATMWARN, NAVFAIL). For each scene and zone we recorded the number of water pixels, the number passing
the mask, and the fraction of pixels flagged for absorbing aerosols (ABSAER), low water-leaving
radiance (LOWLW) and stray light, together with the fraction of masked pixels with negative Rrs at
412 and 443 nm. These diagnostics quantify the atmospheric-correction difficulty of each zone
(Higa et al., 2024).

### 3.2 In-situ
Hourly water-quality records (temperature, salinity, chlorophyll-a, turbidity, dissolved oxygen, pH)
from the MLIT Kanto Regional Development Bureau monitoring posts in inner Tokyo Bay [stations] and the
Japan Coast Guard Chiba light-beacon profile were used for Tokyo Bay. For Sagami Bay, monthly
measurements at the Kanagawa Prefecture public-water-body stations were used. [AERONET-OC Tokyo Bay
SeaPRISM Rrs and co-located chlorophyll-a were used to separate atmospheric-correction error from
in-water algorithm error.] Only surface or upper-layer values (depth ≤ 2 m) were paired with satellite data.

### 3.3 Matchups
Satellite–in-situ pairs were formed within ±3 h, using the median of a 3 × 3 pixel window centred on
the nearest pixel, with the Bailey and Werdell (2006) criteria: at least five valid pixels, values
within 1.5 standard deviations of the window median, and coefficient of variation ≤ 0.15. Matchups
within [1.5] km of land were flagged for adjacency effects. This yielded [N_T] matchups in Tokyo Bay
and [N_S] in Sagami Bay for chlorophyll-a, [N_T′ / N_S′] for turbidity.

## 4. Methods

### 4.1 Baselines
Accuracy is reported at three tiers. Tier 1: standard algorithms with published coefficients (OC4,
Hu et al. 2012 colour index, NDCI, Nechad et al. 2010 TSM, Dogliotti et al. 2015 turbidity) and the
standard PACE and SGLI Level-2 products. Tier 2: the same algorithm forms refitted on the project's
own matchups; the ≥ 20 % RMSE reduction target is defined relative to this tier. Tier 3: the project
models described below.

### 4.2 Optical water types
Rrs spectra (400–720 nm) were area-normalised and clustered with k-means; the number of classes was
selected by silhouette score within 3–7. Each pixel receives a class and a distance to its class centre;
pixels beyond the 95th percentile of training distances, or in a class absent from the training data,
are flagged as novel. The lab's 2020 optical classification of Tokyo Bay is the reference for naming
and comparing the classes.

### 4.3 Retrieval with uncertainty
A random-forest regressor was trained on log10 Rrs resampled to a 10 nm grid, three band ratios and
the optical water type, predicting log10 of the target parameter. The central 68 % interval of each
prediction is the 16th–84th percentile of the per-tree predictions. Uncertainty is validated by its
coverage: the fraction of in-situ values that fall inside the 68 % interval (PICP), reported next to
RMSE, bias, MAPE and median symmetric accuracy (MdSA, Morley et al. 2018).

### 4.4 Transfer experiments
Temporal transfer: models trained on [2024–2025] were tested on [2026], overall and by season. The
adaptive configuration (model refitted at the start of each month with all matchups observed so far)
was compared with the static configuration frozen on the training years. Spatial transfer: models
trained on Tokyo Bay were tested on Sagami Bay and vice versa; errors were split between pixels of
optical types seen in training and novel types. Learning curves were obtained by adding n = 0, 5, 10,
20, 40, 80 randomly chosen target-bay matchups to the training set (five repeats) and evaluating on
the remainder. Spatial block cross-validation (by station) was used within each bay.

### 4.5 Surface–bottom coupling (RQ3 groundwork)
Daily means of surface chlorophyll-a and bottom dissolved oxygen at each monitoring post were
cross-correlated for lags of 0–30 days over May–October. Hypoxia events were defined as ≥ 3
consecutive days with bottom DO < 3 mg L⁻¹ [or the class threshold of the bottom-DO environmental
standard, 2.0 / 3.0 / 4.0 mg L⁻¹]; lead time is the interval between the last surface chlorophyll peak
(above the station's 90th percentile) within 30 days before onset and the onset itself.

## 5. What the first run should show (expected figures)

1. Clear PACE scenes per month, inner Tokyo Bay vs open Sagami Bay — the data budget.
2. Fraction of usable pixels and of ABSAER / negative-blue-Rrs pixels per zone — where the atmospheric correction struggles.
3. Median Rrs spectra per zone on clear days, absolute and shape-normalised — the optical gradient.
4. Standard chlorophyll-a at monitoring posts vs in-situ — tier-1 accuracy before any modelling.

## References (short)
Bailey & Werdell 2006, RSE 102:12 · Higa et al. 2024, Remote Sensing 16:3626 · Hu, Lee & Franz 2012, JGR 117 ·
Mishra & Mishra 2012, RSE 117:394 · Morley, Brito & Welling 2018, Space Weather 16:69 · Nechad et al. 2010, RSE 114:854 ·
Dogliotti et al. 2015, RSE 156:157 · O'Reilly et al. 1998, JGR 103:24937 · Salem et al. 2025, Remote Sensing 17:221
