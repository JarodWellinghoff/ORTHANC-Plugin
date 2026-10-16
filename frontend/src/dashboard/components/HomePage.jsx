import { useNavigate } from "react-router-dom";
import { Box, Button, Card, CardContent, Stack, Typography } from "@mui/material";
import { alpha } from "@mui/material/styles";
import BiotechRoundedIcon from "@mui/icons-material/BiotechRounded";
import AssessmentRoundedIcon from "@mui/icons-material/AssessmentRounded";
import LocalHospitalRoundedIcon from "@mui/icons-material/LocalHospitalRounded";
import SpeedRoundedIcon from "@mui/icons-material/SpeedRounded";
import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
const capabilities = [
  { title: "CHO Detectability", description: "Channelized Hotelling Observer model computes d-prime for low-contrast lesion detectability across reconstructions.", icon: BiotechRoundedIcon },
  { title: "Noise & Resolution", description: "Noise Power Spectrum (NPS) and presampling MTF, including MTF50 and MTF10, measured directly from each CT series.", icon: AssessmentRoundedIcon },
  { title: "Dose Metrics", description: "CTDIvol, body-part-aware SSDE, and DLP recorded per series alongside detectability for full quality and dose context.", icon: LocalHospitalRoundedIcon },
  { title: "End-to-End Pipeline", description: "Pulls from Orthanc, runs analysis with caching, persists to PostgreSQL, and serves results to the dashboard in real time.", icon: SpeedRoundedIcon },
];
export default function HomePage() {
  const navigate = useNavigate();
  return <Stack spacing={5} sx={{ pb: 4 }}>
    <Box sx={(theme) => ({ px: { xs: 3, md: 6 }, py: { xs: 4, md: 7 }, borderRadius: 3,
      border: "1px solid", borderColor: "divider",
      backgroundImage: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.14)} 0%, ${alpha(theme.palette.primary.dark, 0.04)} 100%)` })}>
      <Stack spacing={2.5} sx={{ maxWidth: 1100 }}>
        <Typography variant='h3' component='h1' sx={{ fontWeight: 600, letterSpacing: "-0.02em", fontSize: { xs: "2rem", md: "2.75rem" } }}>
          CTPro: Image Quality and Dose Monitoring for Patient CT Exams
        </Typography>
        <Typography variant='body1' color='text.secondary'>Quantify lesion detectability, noise, resolution, and dose across your CT series.</Typography>
        <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
          <Button variant='contained' size='large' endIcon={<ArrowForwardRoundedIcon />} onClick={() => navigate("/main-dashboard")}>Go to Batch Analysis</Button>
          <Button variant='outlined' size='large' onClick={() => navigate("/results")}>View Results</Button>
          <Button variant='outlined' size='large' onClick={() => navigate("/dicom-pulls")}>Pull DICOM Studies</Button>
        </Stack>
      </Stack>
    </Box>
    <Box>
      <Typography variant='h5' sx={{ fontWeight: 600, mb: 3 }}>What this platform measures</Typography>
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", lg: "repeat(4, 1fr)" }, gap: 2 }}>
        {capabilities.map(({ title, description, icon: Icon }) => <Card key={title} variant='outlined'><CardContent><Stack spacing={1.5}>
          <Icon color='primary' sx={{ fontSize: 26 }} /><Typography variant='subtitle1' sx={{ fontWeight: 600 }}>{title}</Typography>
          <Typography variant='body2' color='text.secondary'>{description}</Typography>
        </Stack></CardContent></Card>)}
      </Box>
    </Box>
  </Stack>;
}
