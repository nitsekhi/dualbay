# 3D prototype of the twin (synthetic data)

`dualbay-3d.html` is a single-file Three.js page: bathymetry, daily surface Chl-a with
confidence shown as transparency, bottom-hypoxia volume, stations with satellite-vs-buoy
mismatch, a gradient Hovmöller diagram and station time series for one synthetic year.

**All numbers in it are synthetic** (generated in the page itself, same generator logic as
`dualbay demo`); coastline and bathymetry are hand-drawn. It exists to show the layers and the
interaction of the digital twin before real PACE and in-situ tables arrive. The data structure
(cell × day: chl, uncertainty, clear flag, bottom DO) matches the SQLite store, so real tables
replace the synthetic ones without changing the scene.

Open the file in a browser (needs internet access for the Three.js CDN).
