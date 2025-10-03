import * as React from "react";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

const formatNumber = (value, decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toFixed(decimals);
};

const MetadataSections = ({ data }) => {
  const patient = data?.patient ?? data;
  const study = data?.study ?? data;
  const series = data?.series ?? data;
  const scanner = data?.scanner ?? data;
  const ct = data?.ct ?? data;
  const results = data?.results ?? data;

  const sections = React.useMemo(
    () => [
      {
        title: "Patient Information",
        items: [
          { label: "Patient Name", value: patient?.patient_name ?? "Anonymous" },
          { label: "Patient ID", value: data?.patient_id ?? "Anonymous" },
          { label: "Patient Sex", value: patient?.sex ?? "Anonymous" },
          { label: "Birth Date", value: patient?.birth_date ?? "Anonymous" },
        ],
      },
      {
        title: "Results",
        items: [
          { label: "Average CTDI", value: formatNumber(results?.ctdivol_avg), unit: "mGy" },
          { label: "SSDE", value: formatNumber(results?.ssde), unit: "mGy" },
          { label: "DLP", value: formatNumber(results?.dlp), unit: "mGy*cm" },
          { label: "Average Dw", value: formatNumber(results?.dw_avg), unit: "cm" },
          { label: "Average Noise Level", value: formatNumber(results?.average_noise_level), unit: "HU" },
          { label: "Peak Frequency", value: formatNumber(results?.peak_frequency, 3), unit: "cm^-1" },
          { label: "Average Frequency", value: formatNumber(results?.average_frequency, 3), unit: "cm^-1" },
          { label: "10% Peak Frequency", value: formatNumber(results?.percent_10_frequency, 3), unit: "cm^-1" },
          { label: "DLP*SSDE", value: formatNumber(results?.dlp_ssde, 3) },
          { label: "Average Index of Detectability", value: formatNumber(results?.average_index_of_detectability, 3) },
        ],
      },
      {
        title: "Study Information",
        items: [
          { label: "Study ID", value: study?.study_id ?? "N/A" },
          { label: "Description", value: study?.study_description ?? "N/A" },
          { label: "Institution", value: study?.institution_name ?? "N/A" },
          { label: "Study Date", value: study?.study_date ?? "N/A" },
          { label: "Study Time", value: study?.study_time ?? "N/A" },
        ],
      },
      {
        title: "Series Information",
        items: [
          { label: "Description", value: series?.series_description ?? "N/A" },
          { label: "Body Part Examined", value: series?.body_part_examined ?? "N/A" },
          { label: "Convolution Kernel", value: series?.convolution_kernel ?? "N/A" },
          { label: "Image Count", value: formatNumber(series?.image_count, 0) },
          { label: "Modality", value: series?.modality ?? "N/A" },
          { label: "Patient Position", value: series?.patient_position ?? "N/A" },
          {
            label: "Pixel Spacing",
            value: Array.isArray(series?.pixel_spacing_mm)
              ? `[${series.pixel_spacing_mm[0]}, ${series.pixel_spacing_mm[1]}]`
              : "N/A",
            unit: "mm",
          },
          { label: "Protocol Name", value: series?.protocol_name ?? "N/A" },
          { label: "Rows", value: formatNumber(series?.rows, 0) },
          { label: "Scan Length", value: formatNumber(series?.scan_length_cm, 3), unit: "cm" },
          { label: "Series Date", value: series?.series_date ?? "N/A" },
          { label: "Series Number", value: series?.series_number ?? "N/A" },
          { label: "Series Time", value: series?.series_time ?? "N/A" },
          { label: "Slice Thickness", value: formatNumber(series?.slice_thickness_mm, 3), unit: "mm" },
          { label: "Columns", value: formatNumber(series?.columns, 0) },
        ],
      },
      {
        title: "Scanner Information",
        items: [
          { label: "Serial Number", value: scanner?.serial_number ?? "N/A" },
          { label: "Institution", value: scanner?.institution_name ?? "N/A" },
          { label: "Manufacturer", value: scanner?.manufacturer ?? "N/A" },
          { label: "Model Name", value: scanner?.model_name ?? "N/A" },
          { label: "Station Name", value: scanner?.station_name ?? "N/A" },
        ],
      },
      {
        title: "CT Technical Parameters",
        items: [
          { label: "Data Collection Diameter", value: formatNumber(ct?.data_collection_diam_mm, 3), unit: "mm" },
          { label: "Distance: Source to Detector", value: formatNumber(ct?.dist_src_detector_mm, 3), unit: "mm" },
          { label: "Distance: Source to Patient", value: formatNumber(ct?.dist_src_patient_mm, 3), unit: "mm" },
          { label: "Exposure Type", value: data?.exposure_modulation_type ?? "N/A" },
          { label: "Exposure Time", value: formatNumber(ct?.exposure_time_ms, 0), unit: "ms" },
          { label: "Filter Type", value: ct?.filter_type ?? "N/A" },
          { label: "Focal Spots", value: ct?.focal_spots_mm ?? "N/A", unit: "mm" },
          { label: "Gantry Angle", value: formatNumber(ct?.gantry_detector_tilt_deg, 3), unit: "deg" },
          { label: "Generator Power", value: formatNumber(ct?.generator_power_kw, 3), unit: "kW" },
          { label: "kvp", value: formatNumber(ct?.kvp, 3), unit: "kV" },
          { label: "Recon Diameter", value: formatNumber(ct?.recon_diameter_mm, 3), unit: "mm" },
          { label: "Single Collimation Width", value: formatNumber(ct?.single_collimation_width_mm, 3), unit: "mm" },
          { label: "Spiral Pitch Factor", value: formatNumber(ct?.spiral_pitch_factor, 3) },
          { label: "Table Feed Per Rotation", value: formatNumber(ct?.table_feed_per_rot_mm, 3), unit: "mm" },
          { label: "Table Speed", value: formatNumber(ct?.table_speed_mm_s, 3), unit: "mm/s" },
          { label: "Total Collimation Width", value: formatNumber(ct?.total_collimation_width_mm, 3), unit: "mm" },
        ],
      },
    ],
    [patient, data, results, study, series, scanner, ct]
  );

  return (
    <Stack spacing={2} divider={<Divider flexItem />}>
      {sections.map((section) => {
        const visibleItems = section.items.filter((item) => item.value !== "N/A");
        if (visibleItems.length === 0) return null;
        return (
          <Box key={section.title}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
              {section.title}
            </Typography>
            <Stack spacing={0.5}>
              {visibleItems.map((item) => (
                <Stack key={item.label} direction="row" spacing={1} justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    {item.label}
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {item.value}
                    {item.unit ? <span style={{ marginLeft: 4 }}>{item.unit}</span> : null}
                  </Typography>
                </Stack>
              ))}
            </Stack>
          </Box>
        );
      })}
    </Stack>
  );
};

export default MetadataSections;
