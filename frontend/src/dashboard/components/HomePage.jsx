import { useNavigate } from "react-router-dom";
import {
  Box,
  Button,
  Card,
  CardContent,
  Stack,
  Typography,
} from "@mui/material";
import { alpha } from "@mui/material/styles";
import ContentPasteSearchIcon from "@mui/icons-material/ContentPasteSearch";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import ScienceRoundedIcon from "@mui/icons-material/ScienceRounded";
import InsightsRoundedIcon from "@mui/icons-material/InsightsRounded";
import OpenInNewRoundedIcon from "@mui/icons-material/OpenInNewRounded";
const capabilities = [
  //   {
  //     title: "CHO Detectability",
  //     description:
  //       "Channelized Hotelling Observer model computes d-prime for low-contrast lesion detectability across reconstructions.",
  //     icon: BiotechRoundedIcon,
  //   },
  //   {
  //     title: "Noise & Resolution",
  //     description:
  //       "Noise Power Spectrum (NPS) and presampling MTF, including MTF50 and MTF10, measured directly from each CT series.",
  //     icon: AssessmentRoundedIcon,
  //   },
  //   {
  //     title: "Dose Metrics",
  //     description:
  //       "CTDIvol, body-part-aware SSDE, and DLP recorded per series alongside detectability for full quality and dose context.",
  //     icon: LocalHospitalRoundedIcon,
  //   },
];
const mainNavItems = [
  {
    text: "DICOM Query and Pulls",
    icon: <CloudDownloadRoundedIcon />,
    to: "/dicom-pulls",
  },
  {
    text: "Batch Analysis",
    icon: <ScienceRoundedIcon />,
    to: "/main-dashboard",
  },
  {
    text: "Result View/Modification of Individual Exam",
    icon: <ContentPasteSearchIcon />,
    to: "/results",
  },
  { text: "Statistics Report", icon: <InsightsRoundedIcon />, to: "/plotting" },
];
export default function HomePage() {
  const navigate = useNavigate();
  return (
    <Stack spacing={5} sx={{ pb: 4 }}>
      <Box
        sx={(theme) => ({
          px: { xs: 3, md: 6 },
          py: { xs: 4, md: 7 },
          borderRadius: 3,
          border: "1px solid",
          borderColor: "divider",
          backgroundImage: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.14)} 0%, ${alpha(theme.palette.primary.dark, 0.04)} 100%)`,
        })}>
        <Stack spacing={2.5} sx={{ maxWidth: 1300 }}>
          <Typography
            variant='h3'
            component='h1'
            sx={{
              fontWeight: 600,
              letterSpacing: "-0.02em",
              fontSize: { xs: "2rem", md: "2.75rem" },
            }}>
            CTPro: Image Quality and Dose Monitoring for Patient CT Exams
          </Typography>
          <Typography variant='body1' color='text.secondary'>
            Measure and monitor lesion detectability, noise, and dose for
            individual patient CT exams
          </Typography>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
            {mainNavItems.map(({ text, to, icon }) => (
              <Button
                key={text}
                variant='outlined'
                size='large'
                startIcon={icon}
                onClick={() => navigate(to)}>
                {text}
              </Button>
            ))}
          </Stack>
        </Stack>
      </Box>
      <Box component='section' aria-labelledby='home-measures-heading'>
        <Box>
          <Stack spacing={1} sx={{ mb: 3 }}>
            <Typography
              id='home-measures-heading'
              variant='h5'
              component='h2'
              sx={{ fontWeight: 600 }}>
              What this platform measures
            </Typography>
          </Stack>

          <Box
            component='figure'
            sx={(theme) => ({
              m: 0,
              overflow: "hidden",
              borderRadius: 3,
              border: "1px solid",
              borderColor: "divider",
              bgcolor: "background.paper",
              boxShadow: `0 16px 48px ${alpha(
                theme.palette.common.black,
                theme.palette.mode === "dark" ? 0.3 : 0.1,
              )}`,
            })}>
            {/* A subtle frame separates the screenshot from the page. */}
            <Box
              sx={(theme) => ({
                p: { xs: 1, sm: 2 },
                backgroundImage: `linear-gradient(
            135deg,
            ${alpha(theme.palette.primary.main, 0.1)} 0%,
            ${alpha(theme.palette.primary.main, 0.02)} 100%
          )`,
              })}>
              <Box
                component='a'
                href='/homepage.png'
                target='_blank'
                rel='noopener noreferrer'
                aria-label='Open example CT analysis results at full size in a new tab'
                sx={(theme) => ({
                  display: "block",
                  overflow: "hidden",
                  borderRadius: 1.5,
                  cursor: "zoom-in",
                  "&:focus-visible": {
                    outline: `3px solid ${theme.palette.primary.main}`,
                    outlineOffset: 4,
                  },
                })}>
                <Box
                  component='img'
                  src='/homepage.png'
                  alt='Example CT analysis results showing image-quality plots and summary measurements.'
                  decoding='async'
                  sx={{
                    display: "block",
                    width: "100%",
                    height: "auto",
                  }}
                />
              </Box>
            </Box>

            {/* <Stack
              component='figcaption'
              direction={{ xs: "column", sm: "row" }}
              spacing={2}
              sx={{
                px: { xs: 2, sm: 3 },
                py: 2,
                alignItems: { xs: "flex-start", sm: "center" },
                justifyContent: "space-between",
                borderTop: "1px solid",
                borderColor: "divider",
              }}>
              <Box sx={{ minWidth: 0 }}>
                <Typography
                  variant='subtitle2'
                  component='p'
                  sx={{ fontWeight: 600 }}>
                  Example analysis results
                </Typography>

                <Typography
                  variant='body2'
                  component='p'
                  color='text.secondary'
                  sx={{ mt: 0.5 }}>
                  Image-quality plots and scan measurements in one view.
                </Typography>
              </Box>

              <Button
                component='a'
                href='/homepage.png'
                target='_blank'
                rel='noopener noreferrer'
                variant='outlined'
                size='small'
                endIcon={<OpenInNewRoundedIcon />}
                aria-label='Open full-size example results image in a new tab'
                sx={{
                  flexShrink: 0,
                  whiteSpace: "nowrap",
                }}>
                Open full size
              </Button>
            </Stack> */}
          </Box>
        </Box>
      </Box>
      {/* <Box>
        <Typography variant='h5' sx={{ fontWeight: 600, mb: 3 }}>
          What this platform measures
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: "1fr",
              sm: "repeat(2, 1fr)",
              lg: "repeat(3, 1fr)",
            },
            gap: 2,
          }}>
          {capabilities.map(({ title, description, icon: Icon }) => (
            <Card key={title} variant='outlined'>
              <CardContent>
                <Stack spacing={1.5}>
                  <Icon color='primary' sx={{ fontSize: 26 }} />
                  <Typography variant='subtitle1' sx={{ fontWeight: 600 }}>
                    {title}
                  </Typography>
                  <Typography variant='body2' color='text.secondary'>
                    {description}
                  </Typography>
                </Stack>
              </CardContent>
            </Card>
          ))}
          <img
            style={{ width: "300px", height: "auto" }}
            src='/homepage.png'
            alt='Example of Results'
          />
        </Box>
      </Box> */}
    </Stack>
  );
}
