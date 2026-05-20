---
title: "Copernicus Emergency Management Service — Rapid Mapping"
location: "European Union / Global"
year: 2023
type: methodology
source: "Copernicus EMS documentation; ESA Sentinel-2 mission documentation; JRC Technical Reports; copernicus.eu/en/services/emergency-management"
---

The Copernicus Emergency Management Service (Copernicus EMS) is a component of the EU's Copernicus Earth Observation programme, operated by the Joint Research Centre (JRC) on behalf of the European Commission. Its Rapid Mapping component provides satellite-derived geospatial information within hours to days of a disaster event.

**Activation**: Copernicus EMS Rapid Mapping can be activated by EU member state civil protection authorities, EU institutions, and certain international organizations. Activations are catalogued with an EMSR code (e.g., EMSR681 for the 2023 Dadia fire). Each activation is publicly archived at the Copernicus EMS portal.

**Products**: For wildfire activations, standard products include:
- *Delineation product*: perimeter of the burned area derived from satellite imagery
- *Grading product*: burn severity map using dNBR or related spectral indices
- *Damage assessment*: structural damage to buildings and infrastructure

**Imagery sources**: Primary sources are Copernicus Sentinel-1 (SAR, cloud-penetrating) and Sentinel-2 (multispectral, 10m visible/NIR bands, 20m SWIR). Commercial very-high-resolution (VHR) imagery from Pleiades, SPOT, or WorldView is used for urban damage assessment.

**dNBR methodology**: Burn severity grading uses the differenced Normalized Burn Ratio: dNBR = pre-fire NBR − post-fire NBR, where NBR = (NIR − SWIR) / (NIR + SWIR). USGS burn severity thresholds are used: enhanced regrowth (<−0.10), unburned (−0.10 to 0.10), low severity (0.10–0.27), moderate-low (0.27–0.44), moderate-high (0.44–0.66), high severity (>0.66).

**Timeliness**: First emergency products are typically delivered within 12–48 hours of activation. Full grading maps follow within 3–7 days depending on cloud cover and satellite revisit intervals.
