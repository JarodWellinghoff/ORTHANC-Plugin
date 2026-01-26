export const formatNumber = (value, decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toFixed(decimals);
};

export const RESULT_METRICS = [
  //   {
  //     key: "peak_frequency",
  //     label: "NPS Peak Frequency",
  //     unit: "cm⁻¹",
  //     decimals: 3,
  //   },
  {
    key: "peak_frequency",
    label: { main: "NPS Peak Frequency" },
    unit: { main: "cm", post: { sup: "-1" }, decimals: 3 },
  },
  //   {
  //     key: "average_frequency",
  //     label: "NPS Average Frequency",
  //     unit: "cm⁻¹",
  //     decimals: 3,
  //   },
  {
    key: "average_frequency",
    label: { main: "NPS Average Frequency" },
    unit: { main: "cm", post: { sup: "-1" }, decimals: 3 },
  },
  // ! Removed
  //   {
  //     key: "percent_10_frequency",
  //     label: "NPS 10% Peak Frequency",
  //     unit: "cm⁻¹",
  //     decimals: 3,
  //   },
  //   { key: "ctdivol_avg", label: "Average CTDIvol", unit: "mGy" },
  {
    key: "ctdivol_avg",
    label: { main: "Average CTDIvol" },
    unit: { main: "mGy" },
  },
  //   { key: "dw_avg", label: "Average Dw", unit: "cm" },
  {
    key: "dw_avg",
    label: { main: "Average D", post: { sub: "w" } },
    unit: { main: "cm" },
  },
  //   { key: "ssde", label: "Average SSDE", unit: "mGy" },
  {
    key: "ssde",
    label: { main: "Average SSDE" },
    unit: { main: "mGy" },
  },
  //   { key: "dlp", label: "DLP (CTDIvol×L)", unit: "mGy*cm" },
  {
    key: "dlp",
    label: { main: "DLP", post: { sub: "CTDIvol" } },
    unit: { main: "mGy•cm" },
    help_text:
      "Dose Length Product calculated using CT Dose Index volume (CTDIvol) multiplied by scan length L.",
  },
  {
    key: "dlp_ssde",
    // label: "DLP_SSDE (SSDE×L)",
    label: { main: "DLP", post: { sub: "SSDE" } },
    unit: { main: "mGy•cm", decimals: 3 },
    help_text:
      "Dose Length Product calculated using Size-Specific Dose Estimate (SSDE) multiplied by scan length L.",
  },
  // ! Removed
  //   {
  //     key: "spatial_resolution",
  //     label: "Spatial Resolution",
  //     unit: "cm⁻¹",
  //     decimals: 1,
  //   },
  //   { key: "average_noise_level", label: "Average Noise Level", unit: "HU" },
  {
    key: "average_noise_level",
    label: { main: "Average Noise Level" },
    unit: { main: "HU" },
  },
  //   {
  //     key: "average_index_of_detectability",
  //     label: "Average Detectability Index ",
  //     decimals: 3,
  //   },
  {
    key: "average_index_of_detectability",
    label: { main: "Average Detectability Index" },
    unit: { decimals: 3 },
  },
];

export const buildMetadataSections = (data) => {
  const patient = data?.patient ?? data;
  const study = data?.study ?? data;
  const series = data?.series ?? data;
  const scanner = data?.scanner ?? data;
  const ct = data?.ct ?? data;
  const results = data?.results ?? data;
  console.log("results", results);
  //   results.average_index_of_detectability = 1.4206697096726497;

  return [
    {
      title: "Patient Information",
      items: [
        { label: "Patient Name", value: patient?.patient_name ?? "Anonymous" },
        { label: "Patient ID", value: patient?.patient_id ?? "Anonymous" },
        { label: "Patient Sex", value: patient?.sex ?? "Anonymous" },
        { label: "Birth Date", value: patient?.birth_date ?? "Anonymous" },
      ],
    },
    {
      title: "Results",
      items: RESULT_METRICS.map(
        ({ key, label, unit, help_text, decimals = 2 }) => ({
          key,
          label,
          value: formatNumber(results?.[key], unit?.decimals ?? decimals),
          unit,
          help_text,
        })
      ),
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
        {
          label: "Body Part Examined",
          value: series?.body_part_examined ?? "N/A",
        },
        {
          label: "Convolution Kernel",
          value: series?.convolution_kernel ?? "N/A",
        },
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
        {
          label: "Scan Length",
          value: formatNumber(series?.scan_length_cm, 3),
          unit: "cm",
        },
        { label: "Series Date", value: series?.series_date ?? "N/A" },
        { label: "Series Number", value: series?.series_number ?? "N/A" },
        { label: "Series Time", value: series?.series_time ?? "N/A" },
        {
          label: "Slice Thickness",
          value: formatNumber(series?.slice_thickness_mm, 3),
          unit: "mm",
        },
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
        {
          label: "Data Collection Diameter",
          value: formatNumber(ct?.data_collection_diam_mm, 3),
          unit: "mm",
        },
        {
          label: "Distance: Source to Detector",
          value: formatNumber(ct?.dist_src_detector_mm, 3),
          unit: "mm",
        },
        {
          label: "Distance: Source to Patient",
          value: formatNumber(ct?.dist_src_patient_mm, 3),
          unit: "mm",
        },
        {
          label: "Exposure Type",
          value: data?.exposure_modulation_type ?? "N/A",
        },
        {
          label: "Exposure Time",
          value: formatNumber(ct?.exposure_time_ms, 0),
          unit: "ms",
        },
        { label: "Filter Type", value: ct?.filter_type ?? "N/A" },
        {
          label: "Focal Spots",
          value: ct?.focal_spots_mm ?? "N/A",
          unit: "mm",
        },
        {
          label: "Gantry Angle",
          value: formatNumber(ct?.gantry_detector_tilt_deg, 3),
          unit: "deg",
        },
        {
          label: "Generator Power",
          value: formatNumber(ct?.generator_power_kw, 3),
          unit: "kW",
        },
        { label: "kvp", value: formatNumber(ct?.kvp, 3), unit: "kV" },
        {
          label: "Recon Diameter",
          value: formatNumber(ct?.recon_diameter_mm, 3),
          unit: "mm",
        },
        {
          label: "Single Collimation Width",
          value: formatNumber(ct?.single_collimation_width_mm, 3),
          unit: "mm",
        },
        {
          label: "Spiral Pitch Factor",
          value: formatNumber(ct?.spiral_pitch_factor, 3),
        },
        {
          label: "Table Feed Per Rotation",
          value: formatNumber(ct?.table_feed_per_rot_mm, 3),
          unit: "mm",
        },
        {
          label: "Table Speed",
          value: formatNumber(ct?.table_speed_mm_s, 3),
          unit: "mm/s",
        },
        {
          label: "Total Collimation Width",
          value: formatNumber(ct?.total_collimation_width_mm, 3),
          unit: "mm",
        },
      ],
    },
  ];
};

export default buildMetadataSections;
